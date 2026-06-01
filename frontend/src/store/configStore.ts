import { create } from 'zustand';

export interface ConfigData {
  LLM_API_KEY: string;
  LLM_BASE_URL: string;
  LLM_MODEL: string;
  LLM_PROVIDER: string;
  LLM_TEMPERATURE: number;
  LLM_MAX_TOKENS: number;
  LLM_MAX_CONCURRENT: number;
  EMBED_MODEL: string;
  EMBED_BASE_URL: string;
  EMBED_MAX_CONCURRENT: number;
  BRAIN_EMAIL: string;
  BRAIN_PASSWORD: string;
  BRAIN_SUBMIT_ENABLED: boolean;
  BRAIN_POLL_TIMEOUT: number;
  AUTOBRAIN_SIM_ENABLED: boolean;
  PIPELINE_MODE: boolean;
  PIPELINE_MAX_SLOTS: number;
  GENERATOR_PARALLEL_TASKS: number;
  MAX_CYCLES: number;
  MAX_MUTATIONS: number;
  LOG_LEVEL: string;
  SUCCESS_CASE_LIBRARY_ENABLED: boolean;
  FAILURE_FIX_LIBRARY_ENABLED: boolean;
  EXPERIENCE_DISTILLER_ENABLED: boolean;
  STRATEGY_CLASSIFIER_ENABLED: boolean;
  FEATURE_MAP_ENABLED: boolean;
  RAG_ENABLED: boolean;
  RAG_TOOL_CALL_ENABLED: boolean;
  RAG_BUDGET_PER_CYCLE: number;
  RAG_TOP_K_OPS: number;
  RAG_TOP_K_FIELDS: number;
  CROSSOVER_ENABLED: boolean;
  MAB_ENABLED: boolean;
  MULTI_AGENT_ENABLED: boolean;
  ORIGINALITY_CHECK_ENABLED: boolean;
  COMPLEXITY_CHECK_ENABLED: boolean;
  EVIDENCE_RECORDING_ENABLED: boolean;
  REFLECTION_ENGINE_ENABLED: boolean;
  TOOL_FACTORY_ENABLED: boolean;
  SEMANTIC_MUTATOR_ENABLED: boolean;
  HYPOTHESIS_ALIGNER_ENABLED: boolean;
  ADAPTIVE_AGENT_ENABLED: boolean;
  MARKET_STATE_ENABLED: boolean;
  PARAM_OPTIMIZATION_ENABLED: boolean;
  SIGNAL_ARBITER_ENABLED: boolean;
  FASTEXPR_GRAMMAR_ENABLED: boolean;
  EVOLUTION_DB_ENABLED: boolean;
  ALPHA_CHANNEL_ENABLED: boolean;
  DIAGNOSIS_LLM_ENABLED: boolean;
  continuous_mode: boolean;
}

interface TestResult {
  success: boolean;
  message: string;
}

interface ConfigStore {
  config: ConfigData;
  originalConfig: ConfigData | null;
  loading: boolean;
  saving: boolean;
  testing: Record<string, boolean>;
  testResults: Record<string, TestResult | null>;
  availableModels: string[];
  loadingModels: boolean;
  loadConfig: () => Promise<void>;
  saveConfig: () => Promise<void>;
  testConnection: (testType: 'llm' | 'brain' | 'embed') => Promise<void>;
  updateField: <K extends keyof ConfigData>(field: K, value: ConfigData[K]) => void;
  resetConfig: () => void;
  fetchModels: () => Promise<void>;
}

const defaultConfig: ConfigData = {
  LLM_API_KEY: '',
  LLM_BASE_URL: 'http://localhost:1234/v1',
  LLM_MODEL: '',
  LLM_PROVIDER: 'lmstudio',
  LLM_TEMPERATURE: 0.7,
  LLM_MAX_TOKENS: 4096,
  LLM_MAX_CONCURRENT: 3,
  EMBED_MODEL: '',
  EMBED_BASE_URL: '',
  EMBED_MAX_CONCURRENT: 3,
  BRAIN_EMAIL: '',
  BRAIN_PASSWORD: '',
  BRAIN_SUBMIT_ENABLED: true,
  BRAIN_POLL_TIMEOUT: 300,
  AUTOBRAIN_SIM_ENABLED: false,
  PIPELINE_MODE: true,
  PIPELINE_MAX_SLOTS: 3,
  GENERATOR_PARALLEL_TASKS: 3,
  MAX_CYCLES: 0,
  MAX_MUTATIONS: 5,
  LOG_LEVEL: 'INFO',
  SUCCESS_CASE_LIBRARY_ENABLED: true,
  FAILURE_FIX_LIBRARY_ENABLED: true,
  EXPERIENCE_DISTILLER_ENABLED: true,
  STRATEGY_CLASSIFIER_ENABLED: true,
  FEATURE_MAP_ENABLED: true,
  RAG_ENABLED: true,
  RAG_TOOL_CALL_ENABLED: true,
  RAG_BUDGET_PER_CYCLE: 3,
  RAG_TOP_K_OPS: 5,
  RAG_TOP_K_FIELDS: 5,
  CROSSOVER_ENABLED: true,
  MAB_ENABLED: true,
  MULTI_AGENT_ENABLED: true,
  ORIGINALITY_CHECK_ENABLED: true,
  COMPLEXITY_CHECK_ENABLED: true,
  EVIDENCE_RECORDING_ENABLED: true,
  REFLECTION_ENGINE_ENABLED: true,
  TOOL_FACTORY_ENABLED: true,
  SEMANTIC_MUTATOR_ENABLED: true,
  HYPOTHESIS_ALIGNER_ENABLED: true,
  ADAPTIVE_AGENT_ENABLED: true,
  MARKET_STATE_ENABLED: true,
  PARAM_OPTIMIZATION_ENABLED: true,
  SIGNAL_ARBITER_ENABLED: true,
  FASTEXPR_GRAMMAR_ENABLED: true,
  EVOLUTION_DB_ENABLED: true,
  ALPHA_CHANNEL_ENABLED: true,
  DIAGNOSIS_LLM_ENABLED: true,
  continuous_mode: true,
};

function flattenBackendConfig(data: Record<string, any>): ConfigData {
  const flat: Record<string, any> = {};
  for (const section of Object.values(data)) {
    if (section && typeof section === 'object') {
      for (const [key, value] of Object.entries(section)) {
        flat[key] = value;
      }
    }
  }
  return { ...defaultConfig, ...flat };
}

export const useConfigStore = create<ConfigStore>((set, get) => ({
  config: { ...defaultConfig },
  originalConfig: null,
  loading: false,
  saving: false,
  testing: { llm: false, brain: false, embed: false },
  testResults: { llm: null, brain: null, embed: null },
  availableModels: [],
  loadingModels: false,

  loadConfig: async () => {
    set({ loading: true });
    try {
      const res = await fetch('/api/config');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const config = flattenBackendConfig(data);
      set({ config, originalConfig: config, loading: false });
      if (config.LLM_PROVIDER === 'lmstudio') {
        get().fetchModels();
      }
    } catch {
      set({ loading: false });
    }
  },

  saveConfig: async () => {
    set({ saving: true });
    try {
      const { continuous_mode, ...backendFields } = get().config;
      const res = await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(backendFields),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      set({ originalConfig: { ...get().config }, saving: false });
    } catch {
      set({ saving: false });
    }
  },

  testConnection: async (testType) => {
    set((s) => ({ testing: { ...s.testing, [testType]: true }, testResults: { ...s.testResults, [testType]: null } }));
    try {
      const res = await fetch('/api/config/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test_type: testType }),
      });
      const data = await res.json();
      const isSuccess = data.status === 'ok' || res.ok;
      const message = data.message || data.detail || (isSuccess ? '连接成功' : '连接失败');
      set((s) => ({
        testing: { ...s.testing, [testType]: false },
        testResults: { ...s.testResults, [testType]: { success: isSuccess, message } },
      }));
    } catch (e) {
      set((s) => ({
        testing: { ...s.testing, [testType]: false },
        testResults: { ...s.testResults, [testType]: { success: false, message: e instanceof Error ? e.message : '连接失败' } },
      }));
    }
  },

  updateField: (field, value) => {
    set((s) => ({ config: { ...s.config, [field]: value } }));
  },

  resetConfig: () => {
    const { originalConfig } = get();
    if (originalConfig) {
      set({ config: { ...originalConfig } });
    }
  },

  fetchModels: async () => {
    set({ loadingModels: true });
    try {
      const res = await fetch('/api/config/models');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const models = Array.isArray(data) ? data : data.models ?? [];
      set({ availableModels: models.map((m: any) => typeof m === 'string' ? m : m.id), loadingModels: false });
    } catch {
      set({ availableModels: [], loadingModels: false });
    }
  },
}));
