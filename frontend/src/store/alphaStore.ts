import { create } from 'zustand'

export type Direction = 'momentum' | 'value' | 'quality' | 'volatility' | 'liquidity' | 'size'
export type AlphaStatus = 'pass' | 'fail' | 'pending'

export interface Alpha {
  id: string
  expression: string
  direction: Direction
  sharpe: number
  fitness: number
  turnover: number
  status: AlphaStatus
  submittedAt: string
  pnlData?: { time: string; value: number }[]
}

export interface AlphaFilters {
  direction: Direction | 'all'
  status: AlphaStatus | 'all'
  sharpeMin: string
  sharpeMax: string
}

interface AlphaState {
  alphas: Alpha[]
  filters: AlphaFilters
  currentAlpha: Alpha | null
  loading: boolean
  error: string | null
  setFilters: (filters: Partial<AlphaFilters>) => void
  resetFilters: () => void
  fetchAlphas: () => Promise<void>
  fetchAlphaDetail: (id: string) => Promise<void>
}

const defaultFilters: AlphaFilters = {
  direction: 'all',
  status: 'all',
  sharpeMin: '',
  sharpeMax: '',
}

export const useAlphaStore = create<AlphaState>((set) => ({
  alphas: [],
  filters: { ...defaultFilters },
  currentAlpha: null,
  loading: false,
  error: null,

  setFilters: (newFilters) =>
    set((state) => ({ filters: { ...state.filters, ...newFilters } })),

  resetFilters: () => set({ filters: { ...defaultFilters } }),

  fetchAlphas: async () => {
    set({ loading: true, error: null })
    try {
      const res = await fetch('/api/alphas')
      if (!res.ok) throw new Error('Failed to fetch alphas')
      const data = await res.json()
      set({ alphas: Array.isArray(data) ? data : data.alphas ?? [], loading: false })
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },

  fetchAlphaDetail: async (id: string) => {
    set({ loading: true, error: null })
    try {
      const res = await fetch(`/api/alphas/${id}`)
      if (!res.ok) throw new Error('Failed to fetch alpha detail')
      const data = await res.json()
      set({ currentAlpha: data, loading: false })
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },
}))
