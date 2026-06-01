import { create } from "zustand";

export type SlotStatus = "idle" | "submitting" | "waiting";
export type GeneratorStatus = "generating" | "validating" | "enqueuing" | "idle";

export interface BrainSlot {
  id: number;
  status: SlotStatus;
  alphaExpression: string | null;
}

export interface MabState {
  initialized: boolean;
  armCount: number;
  health: string;
  arms: { direction: string; alpha: number; beta: number; expectation: number; ucb_score: number }[];
}

export interface Generator {
  id: number;
  direction: string;
  status: GeneratorStatus;
  progress: number;
}

export interface MonitorMetrics {
  avgSharpe: number;
  passRate: number;
  avgFitness: number;
  avgTurnover: number;
}

interface MonitorState {
  slots: BrainSlot[];
  mab: MabState;
  generators: Generator[];
  metrics: MonitorMetrics;
  ws: WebSocket | null;
  connected: boolean;
}

interface MonitorActions {
  fetchMab: () => Promise<void>;
  fetchOverview: () => Promise<void>;
  connectWs: () => void;
  disconnectWs: () => void;
  updateSlot: (index: number, slot: Partial<BrainSlot>) => void;
  updateGenerator: (index: number, gen: Partial<Generator>) => void;
  setMab: (mab: Partial<MabState>) => void;
  setMetrics: (metrics: Partial<MonitorMetrics>) => void;
  reset: () => void;
}

const initialState: MonitorState = {
  slots: [
    { id: 0, status: "idle", alphaExpression: null },
    { id: 1, status: "idle", alphaExpression: null },
    { id: 2, status: "idle", alphaExpression: null },
  ],
  mab: {
    initialized: false,
    armCount: 0,
    health: "unknown",
    arms: [],
  },
  generators: [
    { id: 0, direction: "momentum", status: "idle", progress: 0 },
    { id: 1, direction: "value", status: "idle", progress: 0 },
    { id: 2, direction: "quality", status: "idle", progress: 0 },
  ],
  metrics: {
    avgSharpe: 0,
    passRate: 0,
    avgFitness: 0,
    avgTurnover: 0,
  },
  ws: null,
  connected: false,
};

export const useMonitorStore = create<MonitorState & MonitorActions>((set, get) => ({
  ...initialState,

  fetchMab: async () => {
    try {
      const res = await fetch("/api/status/mab");
      const data = await res.json();
      set({
        mab: {
          initialized: data.initialized ?? false,
          armCount: data.arm_count ?? 0,
          health: data.health ?? "unknown",
          arms: data.arms ?? [],
        },
      });
    } catch {
      // ignore
    }
  },

  fetchOverview: async () => {
    try {
      const res = await fetch("/api/status/overview");
      const data = await res.json();
      set({
        metrics: {
          avgSharpe: data.avg_sharpe ?? 0,
          passRate: data.pass_rate ?? 0,
          avgFitness: data.avg_fitness ?? 0,
          avgTurnover: data.avg_turnover ?? 0,
        },
      });
    } catch {
      // ignore
    }
  },

  connectWs: () => {
    get().disconnectWs();
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/monitor`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => set({ connected: true });

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case "brain_submit": {
            const slotIdx = data.slot ?? data.slot_index;
            if (typeof slotIdx === "number" && slotIdx >= 0 && slotIdx < 3) {
              get().updateSlot(slotIdx, {
                status: "submitting",
                alphaExpression: data.alpha_expression ?? data.expression ?? null,
              });
            }
            break;
          }
          case "brain_result": {
            const slotIdx = data.slot ?? data.slot_index;
            if (typeof slotIdx === "number" && slotIdx >= 0 && slotIdx < 3) {
              get().updateSlot(slotIdx, {
                status: "idle",
                alphaExpression: null,
              });
            }
            break;
          }
          case "mab_update":
            get().setMab({
              initialized: data.initialized ?? get().mab.initialized,
              armCount: data.arm_count ?? get().mab.armCount,
              health: data.health ?? get().mab.health,
              arms: data.arms ?? get().mab.arms,
            });
            break;
          case "generator_update": {
            const genIdx = data.generator ?? data.generator_index;
            if (typeof genIdx === "number" && genIdx >= 0 && genIdx < 3) {
              get().updateGenerator(genIdx, {
                direction: data.direction,
                status: data.status,
                progress: data.progress ?? 0,
              });
            }
            break;
          }
          case "metrics_update":
            get().setMetrics({
              avgSharpe: data.avg_sharpe,
              passRate: data.pass_rate,
              avgFitness: data.avg_fitness,
              avgTurnover: data.avg_turnover,
            });
            break;
        }
      } catch {
        // ignore
      }
    };

    ws.onclose = () => set({ connected: false });
    ws.onerror = () => set({ connected: false });

    set({ ws });
  },

  disconnectWs: () => {
    const { ws } = get();
    if (ws) {
      ws.onclose = null;
      ws.close();
      set({ ws: null, connected: false });
    }
  },

  updateSlot: (index, slot) =>
    set((s) => ({
      slots: s.slots.map((sl, i) => (i === index ? { ...sl, ...slot } : sl)),
    })),

  updateGenerator: (index, gen) =>
    set((s) => ({
      generators: s.generators.map((g, i) => (i === index ? { ...g, ...gen } : g)),
    })),

  setMab: (mab) =>
    set((s) => ({ mab: { ...s.mab, ...mab } })),

  setMetrics: (metrics) =>
    set((s) => ({ metrics: { ...s.metrics, ...metrics } })),

  reset: () => {
    get().disconnectWs();
    set(initialState);
  },
}));
