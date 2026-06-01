import { create } from "zustand";

export type AppMode = "simple" | "pro";
export type WsStatus = "connected" | "disconnected" | "reconnecting";

interface Session {
  id: string;
  status: "idle" | "running" | "paused" | "completed";
  startedAt: string | null;
  alphasFound: number;
}

interface AppState {
  mode: AppMode;
  wsStatus: WsStatus;
  session: Session;

  setMode: (mode: AppMode) => void;
  toggleMode: () => void;
  setWsStatus: (status: WsStatus) => void;
  setSession: (session: Partial<Session>) => void;
  resetSession: () => void;
}

const defaultSession: Session = {
  id: "",
  status: "idle",
  startedAt: null,
  alphasFound: 0,
};

export const useAppStore = create<AppState>((set) => ({
  mode: "simple",
  wsStatus: "disconnected",
  session: { ...defaultSession },

  setMode: (mode) => set({ mode }),
  toggleMode: () =>
    set((state) => ({ mode: state.mode === "simple" ? "pro" : "simple" })),
  setWsStatus: (wsStatus) => set({ wsStatus }),
  setSession: (partial) =>
    set((state) => ({ session: { ...state.session, ...partial } })),
  resetSession: () => set({ session: { ...defaultSession } }),
}));
