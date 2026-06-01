import { useState, useEffect, useRef } from "react";
import { Settings, Bot, Globe, Key, Server, Cpu, Shield, Loader2, CheckCircle2, XCircle, RotateCcw, Save, Eye, EyeOff } from "lucide-react";
import { useConfigStore } from "@/store/configStore";
import { cn } from "@/lib/utils";

type ConfigSection = "llm" | "brain" | "embed" | "pipeline";

const SECTIONS: { key: ConfigSection; label: string; icon: React.ElementType }[] = [
  { key: "llm", label: "LLM 模型", icon: Bot },
  { key: "brain", label: "BRAIN 平台", icon: Globe },
  { key: "embed", label: "Embedding 模型", icon: Cpu },
  { key: "pipeline", label: "流水线参数", icon: Server },
];

const LLM_PROVIDERS = [
  { value: "lmstudio", label: "LM Studio" },
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "groq", label: "Groq" },
  { value: "gemini", label: "Gemini" },
];

function SectionNav({
  active,
  onChange,
}: {
  active: ConfigSection;
  onChange: (s: ConfigSection) => void;
}) {
  return (
    <nav className="flex flex-col gap-1">
      {SECTIONS.map((s) => {
        const Icon = s.icon;
        return (
          <button
            key={s.key}
            onClick={() => onChange(s.key)}
            className={cn(
              "flex items-center gap-3 rounded-m3-lg px-m3-4 py-m3-3 text-label-lg transition-colors text-left",
              active === s.key
                ? "bg-m3-primary/15 text-m3-primary"
                : "text-m3-on-surface-variant hover:bg-m3-surface-container-high"
            )}
          >
            <Icon className="h-5 w-5 shrink-0" />
            {s.label}
          </button>
        );
      })}
    </nav>
  );
}

function TextField({
  label,
  value,
  onChange,
  type = "text",
  placeholder,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <label className="m3-label block">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="m3-input"
      />
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div className="space-y-1.5">
      <label className="m3-label block">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="m3-input"
      />
    </div>
  );
}

function SwitchField({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between py-m3-2">
      <div>
        <div className="text-body-md text-m3-on-surface">{label}</div>
        {description && (
          <div className="text-body-sm text-m3-on-surface-variant mt-0.5">{description}</div>
        )}
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative inline-flex h-8 w-14 shrink-0 rounded-m3-full transition-colors",
          checked ? "bg-m3-primary" : "bg-m3-outline"
        )}
      >
        <span
          className={cn(
            "inline-block h-6 w-6 rounded-full bg-m3-on-primary shadow-sm transition-transform mt-1",
            checked ? "translate-x-7" : "translate-x-1"
          )}
        />
      </button>
    </div>
  );
}

function TestButton({
  testType,
  onTest,
  testing,
  result,
}: {
  testType: "llm" | "brain" | "embed";
  onTest: (t: "llm" | "brain" | "embed") => void;
  testing: boolean;
  result: { success: boolean; message: string } | null;
}) {
  return (
    <div className="flex items-center gap-3 mt-4">
      <button
        onClick={() => onTest(testType)}
        disabled={testing}
        className="m3-btn-outlined flex items-center gap-2"
      >
        {testing ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Shield className="h-4 w-4" />
        )}
        测试连接
      </button>
      {result && (
        <div
          className={cn(
            "flex items-center gap-1.5 text-label-md",
            result.success ? "text-m3-success" : "text-m3-error"
          )}
        >
          {result.success ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <XCircle className="h-4 w-4" />
          )}
          {result.message}
        </div>
      )}
    </div>
  );
}

function ModelCardSelector({
  models,
  selected,
  onSelect,
  loading,
  onRefresh,
}: {
  models: string[];
  selected: string;
  onSelect: (model: string) => void;
  loading: boolean;
  onRefresh: () => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="m3-label block">模型名称</label>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="m3-btn-text text-label-md flex items-center gap-1.5 transition-all hover:scale-[1.02] active:scale-[0.98]"
        >
          {loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Cpu className="h-3.5 w-3.5" />
          )}
          刷新模型列表
        </button>
      </div>

      {models.length > 0 ? (
        <>
          <div
            ref={scrollRef}
            className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin"
            style={{ scrollbarWidth: 'thin' }}
          >
            {models.map((model) => {
              const isSelected = model === selected;
              return (
                <button
                  key={model}
                  onClick={() => onSelect(model)}
                  className={`relative shrink-0 w-48 rounded-m3-md p-4 text-left transition-all duration-300 ${
                    isSelected
                      ? 'glass-strong border-2 border-m3-primary animate-pulseBorder'
                      : 'glass border border-m3-outline-variant hover:border-m3-primary/50 hover:scale-[1.02]'
                  }`}
                >
                  {isSelected && (
                    <div className="absolute -top-1.5 -right-1.5">
                      <div className="flex items-center justify-center rounded-full bg-m3-primary w-5 h-5">
                        <CheckCircle2 className="h-3 w-3 text-m3-on-primary" />
                      </div>
                    </div>
                  )}
                  <div className="text-body-sm font-medium text-m3-on-surface truncate">
                    {model}
                  </div>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <span className="text-label-sm text-m3-on-surface-variant">
                      {model.includes('lmstudio') || model.includes('local') ? '本地' : '云端'}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
          <p className="text-label-sm text-m3-on-surface-variant">
            已检测到 {models.length} 个可用模型 · 点击卡片选择
          </p>
        </>
      ) : (
        <div className="glass rounded-m3-md p-6 text-center">
          <Cpu className="h-8 w-8 text-m3-on-surface-variant/30 mx-auto mb-2" />
          <p className="text-body-sm text-m3-on-surface-variant">未检测到可用模型</p>
          <p className="text-label-sm text-m3-on-surface-variant/60 mt-1">
            请确保 LM Studio 已启动并加载了模型
          </p>
        </div>
      )}

      <div className="flex items-center gap-2">
        <span className="text-label-sm text-m3-on-surface-variant">当前选择:</span>
        <span className="text-body-sm font-mono text-m3-primary">{selected || '未选择'}</span>
      </div>
    </div>
  );
}

function LlmSection() {
  const config = useConfigStore((s) => s.config);
  const updateField = useConfigStore((s) => s.updateField);
  const testing = useConfigStore((s) => s.testing.llm);
  const testResult = useConfigStore((s) => s.testResults.llm);
  const testConnection = useConfigStore((s) => s.testConnection);
  const availableModels = useConfigStore((s) => s.availableModels);
  const fetchModels = useConfigStore((s) => s.fetchModels);
  const loadingModels = useConfigStore((s) => s.loadingModels);

  useEffect(() => {
    if (config.LLM_PROVIDER === "lmstudio") {
      fetchModels();
    }
  }, [config.LLM_PROVIDER, fetchModels]);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="m3-section-title flex items-center gap-2">
          <Bot className="h-5 w-5 text-m3-primary" />
          LLM 模型配置
        </h3>
        <p className="text-body-sm text-m3-on-surface-variant mt-1">
          配置用于Alpha因子生成和改进的大语言模型
        </p>
      </div>

      <div className="m3-card space-y-5">
        <div className="space-y-1.5">
          <label className="m3-label block">提供商</label>
          <select
            value={config.LLM_PROVIDER}
            onChange={(e) => updateField("LLM_PROVIDER", e.target.value)}
            className="m3-select"
          >
            {LLM_PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </div>
        <TextField
          label="Base URL"
          value={config.LLM_BASE_URL}
          onChange={(v) => updateField("LLM_BASE_URL", v)}
          placeholder="http://localhost:1234/v1"
        />
        <TextField
          label="API Key"
          value={config.LLM_API_KEY}
          onChange={(v) => updateField("LLM_API_KEY", v)}
          type="password"
          placeholder="sk-..."
        />
        <ModelCardSelector
          models={availableModels}
          selected={config.LLM_MODEL}
          onSelect={(model) => {
            updateField("LLM_MODEL", model);
            // Auto-save to backend
            setTimeout(() => {
              useConfigStore.getState().saveConfig();
            }, 300);
          }}
          loading={loadingModels}
          onRefresh={fetchModels}
        />
        <div className="space-y-1.5">
          <label className="m3-label block">温度 (Temperature)</label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={config.LLM_TEMPERATURE}
              onChange={(e) => updateField("LLM_TEMPERATURE", Number(e.target.value))}
              className="flex-1 accent-m3-primary"
            />
            <span className="text-body-md text-m3-on-surface w-10 text-right font-mono">
              {config.LLM_TEMPERATURE.toFixed(1)}
            </span>
          </div>
        </div>
        <NumberField
          label="最大 Token 数"
          value={config.LLM_MAX_TOKENS}
          onChange={(v) => updateField("LLM_MAX_TOKENS", v)}
          min={1}
          max={128000}
        />
        <NumberField
          label="最大并发数"
          value={config.LLM_MAX_CONCURRENT}
          onChange={(v) => updateField("LLM_MAX_CONCURRENT", v)}
          min={1}
          max={20}
        />
      </div>

      <TestButton testType="llm" onTest={testConnection} testing={testing} result={testResult} />
    </div>
  );
}

function BrainSection() {
  const [showPassword, setShowPassword] = useState(false);
  const config = useConfigStore((s) => s.config);
  const updateField = useConfigStore((s) => s.updateField);
  const testing = useConfigStore((s) => s.testing.brain);
  const testResult = useConfigStore((s) => s.testResults.brain);
  const testConnection = useConfigStore((s) => s.testConnection);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="m3-section-title flex items-center gap-2">
          <Globe className="h-5 w-5 text-m3-primary" />
          BRAIN 平台配置
        </h3>
        <p className="text-body-sm text-m3-on-surface-variant mt-1">
          配置WorldQuant BRAIN平台的登录凭据，用于Alpha因子提交和验证
        </p>
      </div>

      <div className="m3-card space-y-5">
        <div className="space-y-1.5">
          <label className="m3-label block">邮箱</label>
          <div className="flex items-center gap-2">
            <input
              type="email"
              value={config.BRAIN_EMAIL}
              onChange={(e) => updateField("BRAIN_EMAIL", e.target.value)}
              placeholder="your-email@example.com"
              className="m3-input flex-1"
            />
            {config.BRAIN_EMAIL && (
              <div className="flex items-center gap-1 text-label-sm text-m3-success">
                <CheckCircle2 className="h-3.5 w-3.5" />
                已同步
              </div>
            )}
          </div>
        </div>
        <div className="space-y-1.5">
          <label className="m3-label block">密码</label>
          <div className="relative">
            <input
              type={showPassword ? "text" : "password"}
              value={config.BRAIN_PASSWORD}
              onChange={(e) => updateField("BRAIN_PASSWORD", e.target.value)}
              placeholder="••••••••"
              className="m3-input pr-10"
            />
            <button
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-m3-on-surface-variant hover:text-m3-on-surface transition-colors"
              type="button"
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
        </div>
        <SwitchField
          label="启用自动提交"
          description="自动将通过验证的Alpha因子提交至BRAIN平台"
          checked={config.BRAIN_SUBMIT_ENABLED}
          onChange={(v) => updateField("BRAIN_SUBMIT_ENABLED", v)}
        />
      </div>

      <TestButton testType="brain" onTest={testConnection} testing={testing} result={testResult} />
    </div>
  );
}

function EmbedSection() {
  const config = useConfigStore((s) => s.config);
  const updateField = useConfigStore((s) => s.updateField);
  const testing = useConfigStore((s) => s.testing.embed);
  const testResult = useConfigStore((s) => s.testResults.embed);
  const testConnection = useConfigStore((s) => s.testConnection);

  const isUsingLlmEmbed = config.EMBED_BASE_URL === config.LLM_BASE_URL || !config.EMBED_BASE_URL;

  return (
    <div className="space-y-6">
      <div>
        <h3 className="m3-section-title flex items-center gap-2">
          <Cpu className="h-5 w-5 text-m3-primary" />
          Embedding 模型配置
        </h3>
        <p className="text-body-sm text-m3-on-surface-variant mt-1">
          配置用于语义相似度计算的嵌入模型（RAG检索、FeatureMap等）
        </p>
      </div>

      <div className="m3-card space-y-5">
        <TextField
          label="Base URL"
          value={config.EMBED_BASE_URL}
          onChange={(v) => updateField("EMBED_BASE_URL", v)}
          placeholder="https://api.openai.com/v1"
        />
        <TextField
          label="模型名称"
          value={config.EMBED_MODEL}
          onChange={(v) => updateField("EMBED_MODEL", v)}
          placeholder="text-embedding-3-small"
        />
        {isUsingLlmEmbed && (
          <div className="flex items-center gap-2 rounded-m3-sm bg-m3-primary/5 border border-m3-primary/20 px-3 py-2">
            <CheckCircle2 className="h-4 w-4 text-m3-primary" />
            <span className="text-body-sm text-m3-primary">
              已与 LLM 配置同步: 使用相同的 Base URL ({config.LLM_BASE_URL || '默认'})
            </span>
          </div>
        )}
      </div>

      <TestButton testType="embed" onTest={testConnection} testing={testing} result={testResult} />
    </div>
  );
}

function PipelineSection() {
  const config = useConfigStore((s) => s.config);
  const updateField = useConfigStore((s) => s.updateField);

  const handleContinuousModeChange = (v: boolean) => {
    updateField("continuous_mode", v);
    if (v) {
      updateField("MAX_CYCLES", 0);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="m3-section-title flex items-center gap-2">
          <Server className="h-5 w-5 text-m3-primary" />
          流水线参数
        </h3>
        <p className="text-body-sm text-m3-on-surface-variant mt-1">
          调整Alpha挖掘流水线的运行参数
        </p>
      </div>

      <div className="m3-card space-y-5">
        <SwitchField
          label="循环模式"
          description="7×24小时持续运行，MAX_CYCLES=0 表示无限循环"
          checked={config.continuous_mode}
          onChange={handleContinuousModeChange}
        />
        {!config.continuous_mode && (
          <NumberField
            label="最大循环次数"
            value={config.MAX_CYCLES}
            onChange={(v) => updateField("MAX_CYCLES", v)}
            min={1}
            max={10000}
          />
        )}
        <NumberField
          label="最大变异次数"
          value={config.MAX_MUTATIONS}
          onChange={(v) => updateField("MAX_MUTATIONS", v)}
          min={1}
          max={100}
        />
        <NumberField
          label="BRAIN 并行 Slot 数"
          value={config.PIPELINE_MAX_SLOTS}
          onChange={(v) => updateField("PIPELINE_MAX_SLOTS", v)}
          min={1}
          max={5}
        />
        <NumberField
          label="Generator 并行任务数"
          value={config.GENERATOR_PARALLEL_TASKS}
          onChange={(v) => updateField("GENERATOR_PARALLEL_TASKS", v)}
          min={1}
          max={10}
        />

        <div className="m3-divider" />

        <div className="text-label-lg text-m3-on-surface font-medium">功能开关</div>

        <SwitchField
          label="成功案例库"
          description="记录成功的Alpha因子模式，用于后续生成参考"
          checked={config.SUCCESS_CASE_LIBRARY_ENABLED}
          onChange={(v) => updateField("SUCCESS_CASE_LIBRARY_ENABLED", v)}
        />
        <SwitchField
          label="失败修复库"
          description="记录失败原因和修复策略，避免重复犯错"
          checked={config.FAILURE_FIX_LIBRARY_ENABLED}
          onChange={(v) => updateField("FAILURE_FIX_LIBRARY_ENABLED", v)}
        />
        <SwitchField
          label="经验蒸馏器"
          description="从历史轨迹中蒸馏经验，指导后续探索方向"
          checked={config.EXPERIENCE_DISTILLER_ENABLED}
          onChange={(v) => updateField("EXPERIENCE_DISTILLER_ENABLED", v)}
        />
        <SwitchField
          label="策略分类器"
          description="自动识别和分类Alpha因子的策略类型"
          checked={config.STRATEGY_CLASSIFIER_ENABLED}
          onChange={(v) => updateField("STRATEGY_CLASSIFIER_ENABLED", v)}
        />
        <SwitchField
          label="特征图谱"
          description="构建方向×时间尺度×机制的特征空间映射"
          checked={config.FEATURE_MAP_ENABLED}
          onChange={(v) => updateField("FEATURE_MAP_ENABLED", v)}
        />
        <SwitchField
          label="RAG 检索增强"
          description="基于向量检索历史Alpha因子和经验知识"
          checked={config.RAG_ENABLED}
          onChange={(v) => updateField("RAG_ENABLED", v)}
        />
        <SwitchField
          label="交叉组合"
          description="将不同方向的优秀Alpha因子进行交叉组合"
          checked={config.CROSSOVER_ENABLED}
          onChange={(v) => updateField("CROSSOVER_ENABLED", v)}
        />
        <SwitchField
          label="多臂老虎机 (MAB)"
          description="使用UCB算法动态选择最优探索方向"
          checked={config.MAB_ENABLED}
          onChange={(v) => updateField("MAB_ENABLED", v)}
        />
      </div>
    </div>
  );
}

const SECTION_MAP: Record<ConfigSection, React.ComponentType> = {
  llm: LlmSection,
  brain: BrainSection,
  embed: EmbedSection,
  pipeline: PipelineSection,
};

export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState<ConfigSection>("llm");
  const loadConfig = useConfigStore((s) => s.loadConfig);
  const saveConfig = useConfigStore((s) => s.saveConfig);
  const resetConfig = useConfigStore((s) => s.resetConfig);
  const saving = useConfigStore((s) => s.saving);
  const originalConfig = useConfigStore((s) => s.originalConfig);
  const config = useConfigStore((s) => s.config);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const hasChanges = originalConfig
    ? JSON.stringify(config) !== JSON.stringify(originalConfig)
    : false;

  const ActiveComponent = SECTION_MAP[activeSection];

  return (
    <div className="flex h-full flex-col gap-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Settings className="h-6 w-6 text-m3-primary" />
          <h1 className="text-headline-sm text-m3-on-surface font-medium">配置管理</h1>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={resetConfig} disabled={!hasChanges} className="m3-btn-tonal flex items-center gap-2">
            <RotateCcw className="h-4 w-4" />
            重置
          </button>
          <button onClick={saveConfig} disabled={!hasChanges || saving} className="m3-btn-filled flex items-center gap-2 transition-all hover:scale-[1.02] active:scale-[0.98]">
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
            {saving ? "保存中..." : "保存"}
          </button>
        </div>
      </div>

      <div className="flex gap-6">
        <div className="w-56 shrink-0 glass rounded-m3-md p-m3-2">
          <SectionNav active={activeSection} onChange={setActiveSection} />
        </div>
        <div className="flex-1 min-w-0 overflow-y-auto">
          <ActiveComponent />
        </div>
      </div>
    </div>
  );
}
