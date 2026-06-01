import { create } from "zustand";

export type MiningStatus = "idle" | "running" | "paused" | "stopped";
export type LogLevel = "INFO" | "PASS" | "FAIL" | "WARN" | "DEBUG";
export type LogModule =
  | "IdeaAgent"
  | "FactorAgent"
  | "EvalAgent"
  | "MAB"
  | "Crossover"
  | "Gate"
  | "Mutation"
  | "BRAIN"
  | "Knowledge"
  | "System";

export type SignalColor = "green" | "amber" | "red" | "gray";

export interface SignalLightState {
  brain: SignalColor;
  llm: SignalColor;
  queue: number;
}

export interface BrainSlotInfo {
  used: number;
  total: number;
}

export interface MiningLog {
  id: string;
  timestamp: number;
  level: LogLevel;
  module: LogModule;
  message: string;
  rawMessage?: string;
}

export interface MiningMetrics {
  currentCycle: number;
  totalCycles: number;
  generatedAlpha: number;
  submittedBrain: number;
  passedAlpha: number;
  failedAlpha: number;
  passRate: number;
  brainSlots: BrainSlotInfo;
}

interface MiningParams {
  cycles: number;
  focusArea: string;
}

interface MiningState {
  status: MiningStatus;
  sessionId: string | null;
  forceStopped: boolean;
  metrics: MiningMetrics;
  signalLights: SignalLightState;
  logs: MiningLog[];
  params: MiningParams;
  ws: WebSocket | null;
}

interface MiningActions {
  setParams: (params: Partial<MiningParams>) => void;
  startSession: () => Promise<void>;
  pauseSession: () => Promise<void>;
  resumeSession: () => Promise<void>;
  stopSession: (force?: boolean) => Promise<void>;
  addLog: (log: Omit<MiningLog, "id" | "timestamp">) => void;
  clearLogs: () => void;
  updateMetrics: (metrics: Partial<MiningMetrics>) => void;
  updateSignalLights: (lights: Partial<SignalLightState>) => void;
  connectWs: (sessionId: string) => void;
  disconnectWs: () => void;
  reset: () => void;
}

const MAX_LOGS = 500;

const initialState: MiningState = {
  status: "idle",
  sessionId: null,
  forceStopped: false,
  metrics: {
    currentCycle: 0,
    totalCycles: 0,
    generatedAlpha: 0,
    submittedBrain: 0,
    passedAlpha: 0,
    failedAlpha: 0,
    passRate: 0,
    brainSlots: { used: 0, total: 3 },
  },
  signalLights: {
    brain: "gray",
    llm: "gray",
    queue: 0,
  },
  logs: [],
  params: {
    cycles: 10,
    focusArea: "momentum",
  },
  ws: null,
};

const EVENT_MODULE_MAP: Record<string, LogModule> = {
  cycle_complete: "System",
  alpha_generated: "IdeaAgent",
  alpha_passed: "EvalAgent",
  alpha_failed: "EvalAgent",
  brain_submitted: "BRAIN",
  brain_result: "BRAIN",
  mab_selected: "MAB",
  crossover_applied: "Crossover",
  mutation_applied: "Mutation",
  gate_check: "Gate",
  knowledge_retrieved: "Knowledge",
  session_started: "System",
  session_complete: "System",
  improvement_cycle: "FactorAgent",
  degradation_mode: "System",
};

const EVENT_LEVEL_MAP: Record<string, LogLevel> = {
  cycle_complete: "INFO",
  alpha_generated: "INFO",
  alpha_passed: "PASS",
  alpha_failed: "FAIL",
  brain_submitted: "INFO",
  brain_result: "INFO",
  mab_selected: "INFO",
  crossover_applied: "INFO",
  mutation_applied: "INFO",
  gate_check: "INFO",
  knowledge_retrieved: "INFO",
  session_started: "INFO",
  session_complete: "INFO",
  improvement_cycle: "INFO",
  degradation_mode: "WARN",
};

const WS_LEVEL_MAP: Record<string, LogLevel> = {
  info: "INFO",
  pass: "PASS",
  fail: "FAIL",
  warn: "WARN",
  warning: "WARN",
  debug: "DEBUG",
  error: "FAIL",
};

const WS_MODULE_MAP: Record<string, LogModule> = {
  idea: "IdeaAgent",
  idea_agent: "IdeaAgent",
  factor: "FactorAgent",
  factor_agent: "FactorAgent",
  eval: "EvalAgent",
  eval_agent: "EvalAgent",
  mab: "MAB",
  crossover: "Crossover",
  gate: "Gate",
  mutation: "Mutation",
  brain: "BRAIN",
  knowledge: "Knowledge",
  system: "System",
};

function inferModule(data: Record<string, unknown>): LogModule {
  if (data.module && typeof data.module === "string") {
    const lower = data.module.toLowerCase();
    for (const [key, mod] of Object.entries(WS_MODULE_MAP)) {
      if (lower.includes(key)) return mod;
    }
  }
  if (data.source && typeof data.source === "string") {
    const lower = data.source.toLowerCase();
    for (const [key, mod] of Object.entries(WS_MODULE_MAP)) {
      if (lower.includes(key)) return mod;
    }
  }
  if (data.type && typeof data.type === "string" && EVENT_MODULE_MAP[data.type]) {
    return EVENT_MODULE_MAP[data.type];
  }
  return "System";
}

function inferLevel(data: Record<string, unknown>): LogLevel {
  if (data.level && typeof data.level === "string") {
    return WS_LEVEL_MAP[data.level.toLowerCase()] || "INFO";
  }
  if (data.type && typeof data.type === "string" && EVENT_LEVEL_MAP[data.type]) {
    return EVENT_LEVEL_MAP[data.type];
  }
  return "INFO";
}

export const useMiningStore = create<MiningState & MiningActions>((set, get) => ({
  ...initialState,

  setParams: (params) =>
    set((s) => ({ params: { ...s.params, ...params } })),

  startSession: async () => {
    const { params } = get();
    try {
      const res = await fetch("/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          focus_area: params.focusArea,
        }),
      });
      const data = await res.json();
      const sessionId = data.session_id;
      set({ status: "running", sessionId, forceStopped: false });
      get().updateSignalLights({ brain: "amber", llm: "green" });
      get().connectWs(sessionId);
    } catch {
      get().addLog({ level: "FAIL", module: "System", message: "启动会话失败" });
    }
  },

  pauseSession: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await fetch(`/session/${sessionId}/pause`, { method: "POST" });
      set({ status: "paused" });
    } catch {
      get().addLog({ level: "FAIL", module: "System", message: "暂停会话失败" });
    }
  },

  resumeSession: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await fetch(`/session/${sessionId}/resume`, { method: "POST" });
      set({ status: "running" });
    } catch {
      get().addLog({ level: "FAIL", module: "System", message: "继续会话失败" });
    }
  },

  stopSession: async (force = false) => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await fetch(`/session/${sessionId}/stop`, { method: "POST" });
    } catch {
      // ignore
    }
    get().disconnectWs();
    set({ status: "stopped", forceStopped: force });
    get().updateSignalLights({ brain: "gray", llm: "gray", queue: 0 });
    if (force) {
      get().addLog({ level: "WARN", module: "System", message: "已强制停止挖掘会话" });
    }
  },

  addLog: (log) =>
    set((s) => {
      const newLog: MiningLog = {
        ...log,
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        timestamp: Date.now(),
      };
      const logs = [...s.logs, newLog];
      if (logs.length > MAX_LOGS) logs.splice(0, logs.length - MAX_LOGS);
      return { logs };
    }),

  clearLogs: () => set({ logs: [] }),

  updateMetrics: (metrics) =>
    set((s) => ({ metrics: { ...s.metrics, ...metrics } })),

  updateSignalLights: (lights) =>
    set((s) => ({ signalLights: { ...s.signalLights, ...lights } })),

  connectWs: (sessionId) => {
    get().disconnectWs();
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/session/${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const module = inferModule(data);
        const level = inferLevel(data);

        switch (data.type) {
          case "cycle_complete":
            get().updateMetrics({ currentCycle: data.cycle, totalCycles: data.total_cycles || get().metrics.totalCycles });
            get().addLog({ level, module, message: data.message || `循环完成 #${data.cycle}`, rawMessage: data.message });
            break;
          case "alpha_generated":
            get().updateMetrics({ generatedAlpha: get().metrics.generatedAlpha + 1 });
            get().addLog({ level, module, message: data.message || `Alpha已生成: ${data.alpha_id}`, rawMessage: data.message });
            break;
          case "alpha_passed":
            const newPassed = get().metrics.passedAlpha + 1;
            const newFailed = get().metrics.failedAlpha;
            const total = newPassed + newFailed;
            get().updateMetrics({ passedAlpha: newPassed, passRate: total > 0 ? newPassed / total : 0 });
            get().addLog({ level: "PASS", module, message: data.message || `Alpha通过验证: ${data.alpha_id}`, rawMessage: data.message });
            break;
          case "alpha_failed":
            const newFailed2 = get().metrics.failedAlpha + 1;
            const newPassed2 = get().metrics.passedAlpha;
            const total2 = newPassed2 + newFailed2;
            get().updateMetrics({ failedAlpha: newFailed2, passRate: total2 > 0 ? newPassed2 / total2 : 0 });
            get().addLog({ level: "FAIL", module, message: data.message || `Alpha验证失败: ${data.alpha_id}`, rawMessage: data.message });
            break;
          case "brain_submitted":
            get().updateMetrics({ submittedBrain: get().metrics.submittedBrain + 1 });
            get().updateSignalLights({ brain: "amber" });
            get().addLog({ level, module, message: data.message || `已提交BRAIN平台: ${data.alpha_id}`, rawMessage: data.message });
            break;
          case "brain_result":
            if (data.passed) {
              get().addLog({ level: "PASS", module, message: `BRAIN返回结果: 通过 - ${data.alpha_id}`, rawMessage: data.message });
            } else {
              get().addLog({ level: "FAIL", module, message: `BRAIN返回结果: 未通过 - ${data.alpha_id}`, rawMessage: data.message });
            }
            get().updateSignalLights({ brain: "green" });
            break;
          case "mab_selected":
            get().addLog({ level, module, message: data.message || `MAB选择方向: ${data.direction || ""}`, rawMessage: data.message });
            break;
          case "crossover_applied":
            get().addLog({ level, module, message: data.message || "交叉组合已应用", rawMessage: data.message });
            break;
          case "mutation_applied":
            get().addLog({ level, module, message: data.message || "变异已应用", rawMessage: data.message });
            break;
          case "gate_check":
            get().addLog({ level, module, message: data.message || "门控检查", rawMessage: data.message });
            break;
          case "knowledge_retrieved":
            get().addLog({ level, module, message: data.message || "知识库检索完成", rawMessage: data.message });
            break;
          case "session_started":
            get().addLog({ level, module, message: data.message || "挖掘会话已启动", rawMessage: data.message });
            break;
          case "session_complete":
            set({ status: "stopped" });
            get().addLog({ level, module, message: data.message || "挖掘会话已完成", rawMessage: data.message });
            break;
          case "improvement_cycle":
            get().addLog({ level, module, message: data.message || "改进循环", rawMessage: data.message });
            break;
          case "degradation_mode":
            get().addLog({ level: "WARN", module, message: data.message || "降级模式", rawMessage: data.message });
            break;
          case "log":
            get().addLog({ level, module, message: data.message || "", rawMessage: data.message });
            break;
          default:
            if (data.message) {
              get().addLog({ level, module, message: data.message, rawMessage: data.message });
            }
        }
      } catch {
        // ignore
      }
    };

    ws.onclose = () => {
      const currentStatus = get().status;
      if (currentStatus === "running") {
        set({ status: "stopped" });
      }
      get().updateSignalLights({ brain: "gray", llm: "gray", queue: 0 });
    };

    ws.onerror = () => {
      get().addLog({ level: "FAIL", module: "System", message: "WebSocket连接错误" });
    };

    set({ ws });
  },

  disconnectWs: () => {
    const { ws } = get();
    if (ws) {
      ws.onclose = null;
      ws.close();
      set({ ws: null });
    }
  },

  reset: () => {
    get().disconnectWs();
    set(initialState);
  },
}));
