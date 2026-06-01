import { useRef, useEffect, useState, useCallback } from "react";
import {
  Pickaxe,
  Pause,
  Play,
  Square,
  ChevronDown,
  Terminal,
  Trash2,
  ArrowDown,
  AlertTriangle,
  Wifi,
  WifiOff,
  Cpu,
  Loader2,
  Zap,
  Activity,
  TrendingUp,
  CheckCircle2,
  XCircle,
  Radio,
} from "lucide-react";
import { useMiningStore } from "@/store/miningStore";
import { useConfigStore } from "@/store/configStore";
import type { MiningStatus, LogLevel, LogModule, MiningLog, SignalColor } from "@/store/miningStore";

const FOCUS_AREAS = [
  { value: "momentum", label: "动量" },
  { value: "value", label: "价值" },
  { value: "quality", label: "质量" },
  { value: "volatility", label: "波动率" },
  { value: "liquidity", label: "流动性" },
  { value: "size", label: "规模" },
];

const MODULE_CONFIG: Record<LogModule, { icon: string; color: string; label: string }> = {
  IdeaAgent: { icon: "🧠", color: "text-amber-400", label: "创意" },
  FactorAgent: { icon: "⚙️", color: "text-blue-400", label: "因子" },
  EvalAgent: { icon: "📊", color: "text-purple-400", label: "评估" },
  MAB: { icon: "🎰", color: "text-cyan-400", label: "MAB" },
  Crossover: { icon: "🔄", color: "text-pink-400", label: "交叉" },
  Gate: { icon: "📋", color: "text-orange-400", label: "门控" },
  Mutation: { icon: "🧬", color: "text-green-400", label: "变异" },
  BRAIN: { icon: "📡", color: "text-amber-500", label: "BRAIN" },
  Knowledge: { icon: "📚", color: "text-teal-400", label: "知识" },
  System: { icon: "⚡", color: "text-gray-400", label: "系统" },
};

const LEVEL_CONFIG: Record<LogLevel, { label: string; color: string; bg: string }> = {
  INFO: { label: "INFO", color: "text-gray-400", bg: "bg-gray-400/10" },
  PASS: { label: "PASS", color: "text-green-400", bg: "bg-green-400/10" },
  FAIL: { label: "FAIL", color: "text-red-400", bg: "bg-red-400/10" },
  WARN: { label: "WARN", color: "text-amber-400", bg: "bg-amber-400/10" },
  DEBUG: { label: "DEBUG", color: "text-gray-500", bg: "bg-gray-500/10" },
};

const STATUS_CONFIG: Record<MiningStatus, { label: string; dotClass: string; desc: string }> = {
  idle: { label: "空闲", dotClass: "signal-gray", desc: "等待启动" },
  running: { label: "运行中", dotClass: "signal-green", desc: "正在挖掘Alpha" },
  paused: { label: "已暂停", dotClass: "signal-amber", desc: "挖掘已暂停" },
  stopped: { label: "已停止", dotClass: "signal-red", desc: "挖掘已停止" },
};

const ALL_MODULES: LogModule[] = [
  "IdeaAgent", "FactorAgent", "EvalAgent", "MAB",
  "Crossover", "Gate", "Mutation", "BRAIN",
  "Knowledge", "System",
];

const ALL_LEVELS: LogLevel[] = ["INFO", "PASS", "FAIL", "WARN", "DEBUG"];

function SignalDot({ color, label }: { color: SignalColor; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className={`signal-dot signal-${color}`} />
      <span className="text-label-sm text-m3-on-surface-variant">{label}</span>
    </div>
  );
}

function MetricsCards() {
  const metrics = useMiningStore((s) => s.metrics);
  const status = useMiningStore((s) => s.status);
  const [flashCard, setFlashCard] = useState<string | null>(null);

  useEffect(() => {
    if (metrics.passedAlpha > 0) {
      setFlashCard("passed");
      const t = setTimeout(() => setFlashCard(null), 800);
      return () => clearTimeout(t);
    }
  }, [metrics.passedAlpha]);

  useEffect(() => {
    if (metrics.failedAlpha > 0) {
      setFlashCard("failed");
      const t = setTimeout(() => setFlashCard(null), 600);
      return () => clearTimeout(t);
    }
  }, [metrics.failedAlpha]);

  const cyclePercent = metrics.totalCycles > 0
    ? Math.round((metrics.currentCycle / metrics.totalCycles) * 100)
    : 0;

  const cards = [
    {
      key: "cycle",
      label: "当前循环",
      value: `${metrics.currentCycle}/${metrics.totalCycles || "-"}`,
      icon: Activity,
      color: "text-cyan-400",
      sub: cyclePercent > 0 ? `${cyclePercent}%` : "",
    },
    {
      key: "generated",
      label: "已生成",
      value: metrics.generatedAlpha,
      icon: Zap,
      color: "text-amber-400",
      sub: "",
    },
    {
      key: "submitted",
      label: "已提交",
      value: metrics.submittedBrain,
      icon: TrendingUp,
      color: "text-blue-400",
      sub: metrics.brainSlots ? `${metrics.brainSlots.used}/${metrics.brainSlots.total} 槽位` : "",
    },
    {
      key: "passed",
      label: "已通过",
      value: metrics.passedAlpha,
      icon: CheckCircle2,
      color: "text-green-400",
      sub: metrics.passRate > 0 ? `${(metrics.passRate * 100).toFixed(1)}%` : "",
    },
  ];

  if (status === "idle") return null;

  return (
    <div className="grid grid-cols-4 gap-3">
      {cards.map((card) => {
        const Icon = card.icon;
        let flashClass = "";
        if (flashCard === "passed" && card.key === "passed") flashClass = "metric-card-flash-green animate-success-glow";
        if (flashCard === "failed" && card.key === "passed") flashClass = "metric-card-flash-red animate-fail-shake";

        return (
          <div
            key={card.key}
            className={`metric-card glass rounded-m3-md p-3 transition-all duration-300 ${flashClass}`}
          >
            <div className="flex items-center gap-2.5">
              <div className={`rounded-m3-sm p-1.5 bg-white/5 ${card.color}`}>
                <Icon className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-label-sm text-m3-on-surface-variant truncate">{card.label}</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-title-md font-mono text-m3-on-surface animate-metricPop">
                    {card.value}
                  </span>
                  {card.sub && (
                    <span className="text-label-sm text-m3-on-surface-variant">{card.sub}</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ControlBar() {
  const { status, params, setParams, startSession, pauseSession, resumeSession, stopSession, signalLights } = useMiningStore();
  const config = useConfigStore((s) => s.config);
  const isContinuous = config.continuous_mode;
  const [localCycles, setLocalCycles] = useState(isContinuous ? 0 : params.cycles);
  const [localFocus, setLocalFocus] = useState(params.focusArea);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [launchAnim, setLaunchAnim] = useState(false);

  const isIdle = status === "idle" || status === "stopped";
  const isRunning = status === "running";
  const isPaused = status === "paused";

  const handleStart = useCallback(() => {
    setParams({ cycles: isContinuous ? 0 : localCycles, focusArea: localFocus });
    setLaunchAnim(true);
    setTimeout(() => {
      startSession();
      setLaunchAnim(false);
    }, 400);
  }, [localCycles, localFocus, isContinuous, setParams, startSession]);

  const handleForceStop = useCallback(() => {
    stopSession(true);
  }, [stopSession]);

  const statusCfg = STATUS_CONFIG[status];

  const brainLabel = signalLights.brain === "green" ? "已连接"
    : signalLights.brain === "amber" ? "连接中"
    : signalLights.brain === "red" ? "断开"
    : "未连接";

  const llmLabel = signalLights.llm === "green" ? "就绪"
    : signalLights.llm === "amber" ? "调用中"
    : signalLights.llm === "red" ? "异常"
    : "离线";

  return (
    <div className={`glass rounded-m3-md p-m3-4 transition-all duration-300 ${isRunning ? "animate-pulse-border" : ""}`}>
      <div className="flex flex-wrap items-center gap-m3-4">
        <div className="flex items-center gap-m3-2">
          {isIdle ? (
            <button
              onClick={handleStart}
              className={`m3-btn-filled flex items-center gap-m3-2 px-8 py-3 text-body-lg transition-all hover:scale-105 active:scale-95 ${launchAnim ? "animate-miningLaunch" : ""}`}
            >
              <Pickaxe className="h-5 w-5" />
              开始挖掘
            </button>
          ) : (
            <div className="flex items-center gap-m3-2">
              {isRunning && (
                <button
                  onClick={pauseSession}
                  className="m3-btn-outlined flex items-center gap-m3-2 px-5 py-2.5 transition-all hover:border-m3-primary hover:text-m3-primary hover:scale-[1.02] active:scale-[0.98]"
                >
                  <Pause className="h-4 w-4" />
                  暂停
                </button>
              )}
              {isPaused && (
                <button
                  onClick={resumeSession}
                  className="m3-btn-outlined flex items-center gap-m3-2 px-5 py-2.5 border-m3-primary text-m3-primary transition-all hover:bg-m3-primary/10 hover:scale-[1.02] active:scale-[0.98]"
                >
                  <Play className="h-4 w-4" />
                  继续
                </button>
              )}
              <button
                onClick={handleForceStop}
                className="m3-btn-outlined flex items-center gap-m3-2 px-5 py-2.5 border-red-500/60 text-red-400 transition-all hover:bg-red-500/10 hover:border-red-400 hover:scale-[1.02] active:scale-[0.98]"
              >
                <AlertTriangle className="h-4 w-4" />
                强制停止
              </button>
            </div>
          )}
        </div>

        <div className="flex items-center gap-m3-4 ml-auto">
          <div className="flex items-center gap-m3-2">
            <label className="text-body-sm text-m3-on-surface-variant whitespace-nowrap">循环次数</label>
            {isContinuous ? (
              <div className="m3-input w-24 text-center text-body-sm font-mono text-m3-primary">
                ∞ 持续运行
              </div>
            ) : (
              <input
                type="number"
                min={1}
                max={9999}
                value={localCycles}
                onChange={(e) => setLocalCycles(Number(e.target.value))}
                disabled={!isIdle}
                className="m3-input w-24 font-mono text-body-sm text-center outline-none transition-all focus:border-m3-primary disabled:opacity-50"
              />
            )}
          </div>

          <div className="relative">
            <label className="text-body-sm text-m3-on-surface-variant mr-2 whitespace-nowrap">聚焦方向</label>
            <button
              onClick={() => isIdle && setDropdownOpen(!dropdownOpen)}
              disabled={!isIdle}
              className="m3-input inline-flex items-center gap-1 px-3 py-1.5 text-body-sm transition-all hover:border-m3-primary disabled:opacity-50"
            >
              {FOCUS_AREAS.find(a => a.value === localFocus)?.label || localFocus}
              <ChevronDown className={`h-3.5 w-3.5 transition-transform ${dropdownOpen ? "rotate-180" : ""}`} />
            </button>
            {dropdownOpen && isIdle && (
              <div className="absolute right-0 top-full z-10 mt-1 min-w-[140px] glass-strong rounded-m3-sm py-1 shadow-xl">
                {FOCUS_AREAS.map((area) => (
                  <button
                    key={area.value}
                    onClick={() => {
                      setLocalFocus(area.value);
                      setDropdownOpen(false);
                    }}
                    className={`block w-full px-3 py-1.5 text-left text-body-sm transition-colors hover:bg-m3-primary/10 hover:text-m3-primary ${
                      area.value === localFocus ? "text-m3-primary bg-m3-primary/5" : "text-m3-on-surface"
                    }`}
                  >
                    {area.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-3 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className={`signal-dot ${statusCfg.dotClass}`} />
          <span className="text-body-sm text-m3-on-surface-variant">{statusCfg.label}</span>
          {status !== "idle" && (
            <span className="text-label-sm text-m3-on-surface-variant/60">· {statusCfg.desc}</span>
          )}
        </div>

        <div className="h-3 w-px bg-m3-outline-variant" />

        <div className="flex items-center gap-3">
          {isContinuous && (
            <div className="flex items-center gap-1.5 rounded-m3-full bg-m3-primary/10 px-2.5 py-0.5">
              <span className="text-label-sm text-m3-primary font-medium">7×24 持续模式</span>
            </div>
          )}
          <SignalDot color={signalLights.brain} label={`BRAIN ${brainLabel}`} />
          <SignalDot color={signalLights.llm} label={`LLM ${llmLabel}`} />
          {signalLights.queue > 0 && (
            <div className="flex items-center gap-1.5">
              <div className="flex items-center justify-center rounded-full bg-amber-500/20 px-1.5 py-0.5">
                <span className="text-label-sm text-amber-400 font-mono">{signalLights.queue}</span>
              </div>
              <span className="text-label-sm text-m3-on-surface-variant">排队中</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ModuleTag({ module }: { module: LogModule }) {
  const cfg = MODULE_CONFIG[module];
  return (
    <span className={`inline-flex items-center gap-1 text-body-sm font-medium ${cfg.color}`}>
      <span>{cfg.icon}</span>
      <span className="hidden sm:inline">{module}</span>
    </span>
  );
}

function LevelBadge({ level }: { level: LogLevel }) {
  const cfg = LEVEL_CONFIG[level];
  return (
    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-semibold tracking-wide ${cfg.color} ${cfg.bg}`}>
      {cfg.label}
    </span>
  );
}

function FilterBar({
  moduleFilter,
  setModuleFilter,
  levelFilter,
  setLevelFilter,
  onClear,
  logCount,
  totalCount,
}: {
  moduleFilter: Set<LogModule>;
  setModuleFilter: (v: Set<LogModule>) => void;
  levelFilter: Set<LogLevel>;
  setLevelFilter: (v: Set<LogLevel>) => void;
  onClear: () => void;
  logCount: number;
  totalCount: number;
}) {
  const toggleModule = (m: LogModule) => {
    const next = new Set(moduleFilter);
    if (next.has(m)) next.delete(m);
    else next.add(m);
    setModuleFilter(next);
  };

  const toggleLevel = (l: LogLevel) => {
    const next = new Set(levelFilter);
    if (next.has(l)) next.delete(l);
    else next.add(l);
    setLevelFilter(next);
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5 glass-header px-3 py-2">
      {ALL_MODULES.map((m) => (
        <button
          key={m}
          onClick={() => toggleModule(m)}
          className={`m3-chip text-[11px] transition-all ${
            moduleFilter.has(m) ? "m3-chip-filter-active" : "m3-chip-assist"
          }`}
        >
          {MODULE_CONFIG[m].icon} {MODULE_CONFIG[m].label}
        </button>
      ))}
      <span className="mx-1 h-4 w-px bg-m3-outline-variant" />
      {ALL_LEVELS.map((l) => (
        <button
          key={l}
          onClick={() => toggleLevel(l)}
          className={`m3-chip text-[11px] transition-all ${
            levelFilter.has(l) ? "m3-chip-filter-active" : "m3-chip-assist"
          }`}
        >
          {l}
        </button>
      ))}
      <div className="ml-auto flex items-center gap-2">
        <span className="text-label-sm text-m3-on-surface-variant">
          {logCount}/{totalCount}
        </span>
        <button
          onClick={onClear}
          className="m3-chip m3-chip-assist text-[11px]"
        >
          <Trash2 className="h-3 w-3" />
          清空
        </button>
      </div>
    </div>
  );
}

function LogTerminal() {
  const logs = useMiningStore((s) => s.logs);
  const clearLogs = useMiningStore((s) => s.clearLogs);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [hasNew, setHasNew] = useState(false);
  const [moduleFilter, setModuleFilter] = useState<Set<LogModule>>(new Set());
  const [levelFilter, setLevelFilter] = useState<Set<LogLevel>>(new Set());

  const filteredLogs = logs.filter((log) => {
    if (moduleFilter.size > 0 && !moduleFilter.has(log.module)) return false;
    if (levelFilter.size > 0 && !levelFilter.has(log.level)) return false;
    return true;
  });

  const handleScroll = useCallback(() => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 40;
    setAutoScroll(atBottom);
    if (atBottom) setHasNew(false);
  }, []);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    } else if (logs.length > 0) {
      setHasNew(true);
    }
  }, [logs.length, autoScroll]);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      setAutoScroll(true);
      setHasNew(false);
    }
  }, []);

  return (
    <div className="flex flex-1 flex-col overflow-hidden glass rounded-m3-md">
      <div className="flex items-center gap-m3-2 glass-header px-4 py-2.5 rounded-t-m3-md">
        <Terminal className="h-4 w-4 text-m3-primary" />
        <span className="text-body-md font-medium text-m3-on-surface">实时日志</span>
        <div className="ml-auto flex items-center gap-2">
          {autoScroll ? (
            <span className="text-label-sm text-green-400/70">自动滚动</span>
          ) : (
            <span className="text-label-sm text-m3-on-surface-variant/50">已暂停滚动</span>
          )}
        </div>
      </div>

      <FilterBar
        moduleFilter={moduleFilter}
        setModuleFilter={setModuleFilter}
        levelFilter={levelFilter}
        setLevelFilter={setLevelFilter}
        onClear={clearLogs}
        logCount={filteredLogs.length}
        totalCount={logs.length}
      />

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto bg-m3-surface-container-lowest/80 p-3 font-mono text-body-sm leading-relaxed"
        style={{ minHeight: 0 }}
      >
        {filteredLogs.length === 0 ? (
          <div className="flex h-full min-h-[200px] items-center justify-center text-m3-on-surface-variant/40">
            {logs.length === 0 ? (
              <div className="flex flex-col items-center gap-2">
                <Radio className="h-8 w-8 animate-pulse" />
                <span>等待挖掘事件...</span>
              </div>
            ) : (
              "无匹配日志"
            )}
          </div>
        ) : (
          filteredLogs.map((log) => (
            <LogEntry key={log.id} log={log} />
          ))
        )}
      </div>

      {hasNew && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-20 left-1/2 z-10 -translate-x-1/2 flex items-center gap-1.5 glass-strong rounded-full px-4 py-1.5 text-body-sm font-medium text-m3-primary shadow-lg transition-all hover:scale-105 active:scale-95"
        >
          <ArrowDown className="h-3.5 w-3.5" />
          新消息
        </button>
      )}
    </div>
  );
}

function LogEntry({ log }: { log: MiningLog }) {
  const ts = new Date(log.timestamp).toLocaleTimeString("zh-CN", { hour12: false });
  const levelClass = log.level === "PASS" ? "log-entry-pass"
    : log.level === "FAIL" ? "log-entry-fail"
    : log.level === "WARN" ? "log-entry-warn"
    : "log-entry-info";

  return (
    <div className={`flex items-start gap-2 py-0.5 animate-slide-in ${levelClass}`}>
      <span className="shrink-0 text-gray-500 text-body-sm tabular-nums">{ts}</span>
      <span className="shrink-0">
        <ModuleTag module={log.module} />
      </span>
      <span className="shrink-0">
        <LevelBadge level={log.level} />
      </span>
      <span className={`text-body-sm break-all ${log.level === "DEBUG" ? "text-gray-500" : "text-gray-300"}`}>
        {log.message}
      </span>
    </div>
  );
}

export default function Mining() {
  const status = useMiningStore((s) => s.status);

  useEffect(() => {
    if (status === "running") {
      const interval = setInterval(async () => {
        const sessionId = useMiningStore.getState().sessionId;
        if (!sessionId) return;
        try {
          const res = await fetch(`/session/${sessionId}`);
          const data = await res.json();
          useMiningStore.getState().updateMetrics({
            currentCycle: data.current_cycle ?? data.currentCycle ?? 0,
            totalCycles: data.total_cycles ?? data.totalCycles ?? useMiningStore.getState().metrics.totalCycles,
            generatedAlpha: data.generated_alpha ?? data.generatedAlpha ?? 0,
            submittedBrain: data.submitted_brain ?? data.submittedBrain ?? 0,
            passedAlpha: data.passed_alpha ?? data.passedAlpha ?? 0,
          });
          if (data.brain_slots) {
            useMiningStore.getState().updateMetrics({
              brainSlots: data.brain_slots,
            });
          }
        } catch {
          // ignore
        }
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [status]);

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center gap-m3-3">
        <Pickaxe className="h-7 w-7 text-m3-primary" />
        <h1 className="text-headline-sm font-medium text-m3-on-surface">挖掘控制台</h1>
      </div>

      <ControlBar />

      <MetricsCards />

      <div className="relative flex flex-1 flex-col" style={{ minHeight: 0 }}>
        <LogTerminal />
      </div>
    </div>
  );
}
