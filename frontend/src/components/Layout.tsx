import { Outlet } from "react-router-dom";
import Sidebar from "@/components/Sidebar";
import ModeToggle from "@/components/ModeToggle";
import { useAppStore } from "@/store/appStore";
import { cn } from "@/lib/utils";

export default function Layout() {
  const wsStatus = useAppStore((s) => s.wsStatus);

  const statusColor = {
    connected: "bg-m3-success",
    disconnected: "bg-m3-error",
    reconnecting: "bg-m3-primary animate-pulse",
  }[wsStatus];

  const statusLabel = {
    connected: "已连接",
    disconnected: "未连接",
    reconnecting: "重连中",
  }[wsStatus];

  return (
    <div className="flex h-screen bg-m3-surface-container-lowest overflow-hidden">
      <Sidebar />

      <div className="flex flex-col flex-1 min-w-0">
        <header className="flex items-center justify-between h-12 px-m3-4 border-b border-m3-outline-variant bg-m3-surface shrink-0">
          <div className="flex items-center gap-2">
            <span className="font-mono-title text-title-sm font-semibold text-gradient-orange">
              OpenAlpha
            </span>
          </div>

          <div className="flex items-center gap-3">
            <ModeToggle />

            <div className="flex items-center gap-1.5">
              <span className={cn("h-2 w-2 rounded-full", statusColor)} />
              <span className="text-label-sm text-m3-on-surface-variant">{statusLabel}</span>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-auto p-m3-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
