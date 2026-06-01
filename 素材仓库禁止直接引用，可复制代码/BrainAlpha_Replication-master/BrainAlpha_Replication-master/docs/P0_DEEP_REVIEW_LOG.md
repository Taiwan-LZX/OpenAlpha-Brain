# P0 Deep Review & Debug 紀錄

**日期：** 2026-05-21  
**範圍：** P0 全部改動（llm_client.py, expression_generator.py, repair_engine.py, planner.py, engine.py, config.py, tests）  
**結果：** 115/115 測試通過，14 項問題全部修復

---

## 一、Review 方法論

對每個 P0 改動檔案進行以下檢查：
1. **Edge cases** — 空輸入、None、空 dict、空列表
2. **Error handling** — 是否捕獲所有預期異常
3. **Data flow** — 參數是否正確傳遞到下游
4. **Config wiring** — config 欄位是否被實際使用
5. **Dead code** — 未使用的常數、參數、變數
6. **Test coverage** — 是否有未測試的分支

---

## 二、Critical 修復（5 項）

### 2.1 `llm_client.py` — `chat_completion` 未捕獲 `URLError`

**問題：**
```python
# 原始程式碼（僅捕獲 HTTPError）
try:
    with urllib.request.urlopen(request, timeout=self.model_config.timeout) as response:
        return json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    raw = exc.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"LLM HTTP error: {exc.code} {raw[:400]}") from exc
```

**影響：** DNS 解析失敗、連線被拒、逾時等網路錯誤會以 `URLError` 或 `OSError` 拋出，未被捕獲 → 直接中斷管線。

**修復：**
```python
except urllib.error.URLError as exc:
    raise RuntimeError(f"LLM connection error: {exc.reason}") from exc
except OSError as exc:
    raise RuntimeError(f"LLM network error: {exc}") from exc
```

---

### 2.2 `llm_client.py` — `extract_json_payload` 未捕獲 `JSONDecodeError`

**問題：**
```python
# 原始程式碼
if isinstance(content, str):
    return json.loads(content)  # 若 LLM 回傳非 JSON 字串，直接拋出 JSONDecodeError
```

**影響：** LLM 即使設定了 `response_format: json_object`，仍可能回傳非 JSON 內容（如 markdown code block、純文字）。`JSONDecodeError` 未被捕獲 → 直接中斷管線。

**修復：**
```python
if isinstance(content, str):
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
```

---

### 2.3 `repair_engine.py` — `sao.failed_checks` 未從 `execute()` 參數傳遞

**問題：**
```python
# execute() 接收 failed_checks 參數
def execute(self, sao: SignalAlphaObject, failed_checks: Sequence[str]) -> RepairResult:
    ...
    # 但 _llm_rewrite_expression 使用 sao.failed_checks（預設為 []）
    rewritten = self._llm_rewrite_expression(sao, strategy_name)
```

`_build_rewrite_prompt` 中：
```python
f"Failed checks: {', '.join(sao.failed_checks)}\n"  # 永遠是空字串！
```

**影響：** LLM 收到的 prompt 中 `Failed checks:` 永遠是空的，無法知道具體失敗原因 → rewrite 品質下降。

**修復：**
1. 在 `execute()` 頂部加入 `sao.failed_checks = list(failed_checks)`
2. 修改 `_llm_rewrite_expression` 簽名，接受 `failed_checks` 參數
3. 修改 `_build_rewrite_prompt` 簽名，接受 `failed_checks` 參數並使用

```python
# execute()
sao.failed_checks = list(failed_checks)
...
rewritten = self._llm_rewrite_expression(sao, strategy_name, list(failed_checks))

# _build_rewrite_prompt
f"Failed checks: {', '.join(failed_checks) if failed_checks else 'none'}\n"
```

---

### 2.4 `expression_generator.py` — `LLMGenConfig.cot_enabled` 未檢查

**問題：**
`LLMGenConfig` 定義了 `cot_enabled: bool = True`，但 `_llm_generate_expressions` 從未檢查此欄位。即使使用者設定 `cot_enabled=False`，LLM 仍會執行。

**影響：** config 欄位形同虛設，無法獨立控制 CoT 功能。

**修復：**
```python
def _llm_generate_expressions(self, cell, gate, max_candidates=4):
    if not self.llm_client or not self.llm_gen_config:
        return []
    if not self.llm_gen_config.cot_enabled:  # 新增檢查
        return []
    ...
```

---

### 2.5 兩個 test 檔案 — Mock `or` fallthrough bug

**問題：**
```python
# test_expression_generator.py 和 test_repair_engine.py
def __init__(self, response=None):
    self._response = response or {"expression": "rank(close)", "reasoning": "test"}
```

在 Python 中，空 dict `{}` 是 falsy，所以 `{} or default` 會回傳 `default`。

**影響：** 測試 `MockLLMClient({})` 意圖測試空回應 fallback，但實際上回傳了預設值 → 測試無效。

**修復：**
```python
self._response = response if response is not None else {"expression": "rank(close)", "reasoning": "test"}
```

---

## 三、Design 改進（3 項）

### 3.1 `config.py` — 新增 `rewrite_enabled` 欄位

**問題：** `LLMGenConfig` 只有一個 `enabled` 開關，無法獨立控制 M2 生成和 M4 修復。

**修復：**
```python
@dataclass(frozen=True)
class LLMGenConfig:
    enabled: bool = False
    provider: str = "heuristic"
    diversity_threshold: float = 0.15
    cot_enabled: bool = True
    rewrite_enabled: bool = True  # 新增
```

### 3.2 `engine.py` — 串接 `rewrite_enabled`

**修復：**
```python
# ensure_repair_engine()
llm_client: Optional[Any] = None
if self.agent_cfg.llm_gen.enabled and self.agent_cfg.llm_gen.rewrite_enabled:
    llm_client = LLMClient(self.runtime.model)
```

### 3.3 `expression_generator.py` — `_build_cot_prompt` 空欄位保護

**問題：** 若 `cell.candidate_fields` 為空，`fields_str` 會是空字串，prompt 變成 `Verified fields: `。

**修復：**
```python
fields = cell.candidate_fields[:5]
fields_str = ", ".join(fields) if fields else "close, returns, volume"
```

---

## 四、Cleanup（3 項）

### 4.1 `planner.py` — 移除 dead `api_key` 參數

`_request_plan` 接收 `api_key` 但從未使用（LLMClient 內部透過 `os.getenv` 取得）。

### 4.2 `repair_engine.py` — 移除 dead PHASE 常數

`PHASE_1_LABEL`, `PHASE_2_LABEL`, `PHASE_3_LABEL` 定義後從未使用。

### 4.3 `expression_generator.py` — 移除重複 diversity filter

`_llm_generate_expressions` 內部呼叫 `gate.filter_diverse()` 修改了 gate 的 frontier，然後 `generate_from_cell` 再次呼叫 `filter_diverse()`，導致已加入的候選被判定為「重複」而被過濾掉 → **所有 LLM 候選被丟棄**。

這是前一個 session 發現的 bug，修復方式為移除 `_llm_generate_expressions` 內的 `gate.filter_diverse()` 呼叫，由 `generate_from_cell` 統一處理。

---

## 五、新增測試（+14）

### 5.1 `test_llm_client.py`（新檔案，11 項測試）

| 測試名稱 | 驗證內容 |
|---|---|
| `test_valid_string_content` | `extract_json_payload` 解析正常 JSON 字串 |
| `test_valid_list_content` | `extract_json_payload` 解析 list 格式 content |
| `test_no_choices_raises` | 空 choices 拋出 ValueError |
| `test_choices_not_list_raises` | choices 非 list 拋出 ValueError |
| `test_first_choice_not_dict_raises` | choice 不是 dict 拋出 ValueError |
| `test_content_not_string_or_list_raises` | content 格式不支援拋出 ValueError |
| `test_content_is_none_raises` | content 為 None 拋出 ValueError |
| `test_content_not_jsonable_string` | 非 JSON 字串拋出 ValueError（含 "invalid json"） |
| `test_request_json_passes_temperature_and_model` | `request_json` 正確傳遞參數 |
| `test_request_json_uses_default_model` | 未指定 model 時使用預設值 |
| `test_request_json_response_format_is_json_object` | 自動設定 `response_format: json_object` |

### 5.2 `test_expression_generator.py`（+2 項）

| 測試名稱 | 驗證內容 |
|---|---|
| `test_cot_disabled_uses_templates` | `cot_enabled=False` 時 fallback 到模板 |
| `test_llm_empty_expression_falls_back` | LLM 回傳 `{}` 時 fallback 到模板 |

### 5.3 `test_repair_engine.py`（+1 項）

| 測試名稱 | 驗證內容 |
|---|---|
| `test_llm_rewrite_missing_expression_key_falls_back` | LLM 回傳 `{"reasoning": "..."}`（無 expression key）時 fallback 到 `wrap_expression` |

---

## 六、Bug 追蹤摘要

| # | 檔案 | 行號 | 類型 | 嚴重度 | 狀態 |
|---|---|---|---|---|---|
| 1 | `llm_client.py` | 35 | 未捕獲 URLError | Critical | ✅ 已修復 |
| 2 | `llm_client.py` | 47 | 未捕獲 JSONDecodeError | Critical | ✅ 已修復 |
| 3 | `repair_engine.py` | 144, 191, 215, 240 | failed_checks 未傳遞 | Critical | ✅ 已修復 |
| 4 | `expression_generator.py` | 295 | cot_enabled 未檢查 | Critical | ✅ 已修復 |
| 5 | `test_*.py` | Mock __init__ | `or` fallthrough | Critical | ✅ 已修復 |
| 6 | `config.py` | 55 | 缺少 rewrite_enabled | Medium | ✅ 已修復 |
| 7 | `engine.py` | 364 | 未串接 rewrite_enabled | Medium | ✅ 已修復 |
| 8 | `expression_generator.py` | 270 | 空欄位無保護 | Medium | ✅ 已修復 |
| 9 | `planner.py` | 280 | dead api_key 參數 | Minor | ✅ 已修復 |
| 10 | `repair_engine.py` | 40-42 | dead PHASE 常數 | Minor | ✅ 已修復 |
| 11 | `expression_generator.py` | 345 | 重複 diversity filter | Critical | ✅ 已修復（前 session） |
| 12 | `tests/` | — | 缺少 LLMClient 測試 | Medium | ✅ 已新增 |
| 13 | `tests/` | — | 缺少 edge case 測試 | Medium | ✅ 已新增 |
| 14 | `tests/` | — | Mock 構造器 bug | Critical | ✅ 已修復 |

---

## 七、測試結果

```
======================= 115 passed, 1 warning in 4.62s ========================
```

| 測試檔案 | 測試數 | 狀態 |
|---|---|---|
| `test_alpha101_library.py` | 3 | ✅ |
| `test_expression_generator.py` | 15 (+2) | ✅ |
| `test_llm_client.py` | 11 (new) | ✅ |
| `test_planner.py` | 8 | ✅ |
| `test_progress.py` | 1 | ✅ |
| `test_rag_spec.py` | 16 | ✅ |
| `test_repair_engine.py` | 23 (+1) | ✅ |
| `test_research_logic.py` | 6 | ✅ |
| **總計** | **115** | **✅** |

---

## 八、經驗總結

### 8.1 常見陷阱

1. **Python falsy 陷阱** — `{} or default` 不等於 `{} if {} else default`。空 dict/list 是 falsy，應使用 `is None` 檢查。
2. **重複過濾** — 當多個層級共用同一個 stateful 物件（如 `JaccardDiversityGate`）時，內層的 `filter_diverse` 會污染 outer 層的 frontier。
3. **參數未傳遞** — `execute()` 接收 `failed_checks` 但下游使用 `sao.failed_checks`（預設值），造成資料斷層。
4. **Config 欄位未使用** — 定義了 `cot_enabled` 但從未檢查，形同虛設。

### 8.2 防禦性編程建議

1. **LLM 回應永遠不可信** — 即使設定了 `response_format: json_object`，仍可能回傳非 JSON。所有 JSON 解析都應包 try/except。
2. **網路呼叫永遠可能失敗** — 除了 HTTPError，還要捕獲 URLError 和 OSError。
3. **Config 欄位定義了就應該被使用** — 定期檢查是否有未使用的 config 欄位。
4. **Mock 應該精確模擬邊界情況** — 測試空回應、缺少 key、格式錯誤等。

---

*本紀錄於 2026 年 5 月 21 日完成，涵蓋 P0 全部改動的 deep review 和 debug 過程。*
