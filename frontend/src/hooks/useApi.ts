import { useState, useCallback } from "react";
import { api, ApiError } from "@/utils/api";

interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
}

interface UseApiResult<T> extends UseApiState<T> {
  execute: (...args: unknown[]) => Promise<T | null>;
  reset: () => void;
}

export function useApi<T = unknown>(
  method: "get" | "post" | "put" | "patch" | "delete",
  path: string
): UseApiResult<T> {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(
    async (...args: unknown[]) => {
      setState((prev) => ({ ...prev, loading: true, error: null }));

      try {
        let result: T;

        if (method === "get" || method === "delete") {
          const params = (args[0] as Record<string, string>) ?? undefined;
          result = await api[method]<T>(path, params);
        } else {
          const body = args[0];
          result = await api[method]<T>(path, body);
        }

        setState({ data: result, loading: false, error: null });
        return result;
      } catch (err) {
        const error =
          err instanceof ApiError
            ? err
            : new ApiError(0, "Unknown", String(err));

        setState((prev) => ({ ...prev, loading: false, error }));
        return null;
      }
    },
    [method, path]
  );

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return { ...state, execute, reset };
}
