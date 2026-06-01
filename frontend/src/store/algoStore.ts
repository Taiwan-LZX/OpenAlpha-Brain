import { create } from 'zustand'

export interface MabArm {
  direction: string
  alpha: number
  beta: number
  expectation: number
  ucb_score: number
}

export interface FeatureCell {
  direction: string
  time_horizon: string
  mechanism: string
  elite_count: number
}

export interface StrategyProfile {
  direction: string
  mechanism: string
  time_horizon: string
  sharpe: number
}

export interface DecayRecord {
  alpha_id: string
  direction: string
  decay_level: number
  ewma_sharpe: number
  composite_score: number
  initial_sharpe: number
  peak_sharpe: number
  garch_anomaly: boolean
  consecutive_decay_checks: number
}

interface AlgoState {
  mab: MabArm[]
  featureMap: FeatureCell[]
  strategyProfiles: StrategyProfile[]
  decayBlacklist: string[]
  decayRecords: DecayRecord[]
  loading: boolean
  error: string | null
  fetchMab: () => Promise<void>
  fetchFeatureMap: () => Promise<void>
  fetchStrategy: () => Promise<void>
  fetchDecay: () => Promise<void>
  fetchAll: () => Promise<void>
}

export const useAlgoStore = create<AlgoState>((set) => ({
  mab: [],
  featureMap: [],
  strategyProfiles: [],
  decayBlacklist: [],
  decayRecords: [],
  loading: false,
  error: null,

  fetchMab: async () => {
    set({ loading: true, error: null })
    try {
      const res = await fetch('/api/status/mab')
      if (!res.ok) throw new Error('Failed to fetch MAB data')
      const data = await res.json()
      set({ mab: Array.isArray(data) ? data : data.arms ?? [], loading: false })
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },

  fetchFeatureMap: async () => {
    set({ loading: true, error: null })
    try {
      const res = await fetch('/api/status/feature-map')
      if (!res.ok) throw new Error('Failed to fetch feature map')
      const data = await res.json()
      set({ featureMap: Array.isArray(data) ? data : data.cells ?? [], loading: false })
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },

  fetchStrategy: async () => {
    set({ loading: true, error: null })
    try {
      const res = await fetch('/api/status/strategy')
      if (!res.ok) throw new Error('Failed to fetch strategy data')
      const data = await res.json()
      set({
        strategyProfiles: data.profiles ?? Array.isArray(data) ? data : [],
        loading: false,
      })
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },

  fetchDecay: async () => {
    set({ loading: true, error: null })
    try {
      const res = await fetch('/api/status/decay')
      if (!res.ok) throw new Error('Failed to fetch decay data')
      const data = await res.json()
      set({
        decayBlacklist: data.blacklisted_directions ?? [],
        decayRecords: data.records ?? (Array.isArray(data) ? data : []),
        loading: false,
      })
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },

  fetchAll: async () => {
    set({ loading: true, error: null })
    try {
      const [mabRes, fmRes, stratRes, decayRes] = await Promise.all([
        fetch('/api/status/mab'),
        fetch('/api/status/feature-map'),
        fetch('/api/status/strategy'),
        fetch('/api/status/decay'),
      ])
      const [mab, featureMap, strategy, decay] = await Promise.all([
        mabRes.json(),
        fmRes.json(),
        stratRes.json(),
        decayRes.json(),
      ])
      set({
        mab: Array.isArray(mab) ? mab : mab.arms ?? [],
        featureMap: Array.isArray(featureMap) ? featureMap : featureMap.cells ?? [],
        strategyProfiles: strategy.top_profiles ?? strategy.profiles ?? (Array.isArray(strategy) ? strategy : []),
        decayBlacklist: decay.blacklisted_directions ?? [],
        decayRecords: decay.records ?? (Array.isArray(decay) ? decay : []),
        loading: false,
      })
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },
}))
