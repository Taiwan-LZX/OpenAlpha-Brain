import { useEffect, useRef, useCallback } from "react";
import { useAppStore } from "@/store/appStore";

type WsEventHandler = (data: unknown) => void;

const WS_URL = "ws://localhost:8000/ws/events";
const RECONNECT_INTERVAL = 3000;
const MAX_RECONNECT_ATTEMPTS = 10;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef<Map<string, Set<WsEventHandler>>>(new Map());
  const reconnectCountRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const setWsStatus = useAppStore((s) => s.setWsStatus);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        reconnectCountRef.current = 0;
        setWsStatus("connected");
      };

      ws.onclose = () => {
        setWsStatus("disconnected");
        scheduleReconnect();
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          const eventType: string = parsed.type ?? parsed.event ?? "message";
          const handlers = handlersRef.current.get(eventType);
          if (handlers) {
            handlers.forEach((handler) => handler(parsed));
          }
          const wildcardHandlers = handlersRef.current.get("*");
          if (wildcardHandlers) {
            wildcardHandlers.forEach((handler) => handler(parsed));
          }
        } catch {
          const messageHandlers = handlersRef.current.get("message");
          if (messageHandlers) {
            messageHandlers.forEach((handler) => handler(event.data));
          }
        }
      };

      wsRef.current = ws;
    } catch {
      scheduleReconnect();
    }
  }, [setWsStatus]);

  const scheduleReconnect = useCallback(() => {
    if (reconnectCountRef.current >= MAX_RECONNECT_ATTEMPTS) return;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }

    reconnectCountRef.current += 1;
    setWsStatus("reconnecting");

    reconnectTimerRef.current = setTimeout(() => {
      connect();
    }, RECONNECT_INTERVAL);
  }, [connect, setWsStatus]);

  const subscribe = useCallback((eventType: string, handler: WsEventHandler) => {
    if (!handlersRef.current.has(eventType)) {
      handlersRef.current.set(eventType, new Set());
    }
    handlersRef.current.get(eventType)!.add(handler);

    return () => {
      handlersRef.current.get(eventType)?.delete(handler);
    };
  }, []);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    reconnectCountRef.current = MAX_RECONNECT_ATTEMPTS;
    wsRef.current?.close();
    wsRef.current = null;
    setWsStatus("disconnected");
  }, [setWsStatus]);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return { subscribe, send, disconnect };
}
