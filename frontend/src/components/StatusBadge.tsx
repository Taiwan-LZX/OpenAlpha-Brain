import { cn } from "@/lib/utils";

type StatusType = "running" | "paused" | "stopped" | "pass" | "fail";

interface StatusBadgeProps {
  status: StatusType;
  label?: string;
  pulse?: boolean;
  size?: "sm" | "md";
}

const statusConfig: Record<
  StatusType,
  { bg: string; text: string; dot: string; defaultLabel: string }
> = {
  running: {
    bg: "bg-mine-success/10",
    text: "text-mine-success",
    dot: "bg-mine-success",
    defaultLabel: "运行中",
  },
  paused: {
    bg: "bg-mine-orange/10",
    text: "text-mine-orange",
    dot: "bg-mine-orange",
    defaultLabel: "已暂停",
  },
  stopped: {
    bg: "bg-mine-muted/10",
    text: "text-mine-muted",
    dot: "bg-mine-muted",
    defaultLabel: "已停止",
  },
  pass: {
    bg: "bg-mine-success/10",
    text: "text-mine-success",
    dot: "bg-mine-success",
    defaultLabel: "通过",
  },
  fail: {
    bg: "bg-mine-fail/10",
    text: "text-mine-fail",
    dot: "bg-mine-fail",
    defaultLabel: "失败",
  },
};

export default function StatusBadge({
  status,
  label,
  pulse = false,
  size = "sm",
}: StatusBadgeProps) {
  const config = statusConfig[status];
  const displayLabel = label ?? config.defaultLabel;
  const sizeClasses = size === "sm" ? "text-xs px-2 py-0.5" : "text-sm px-3 py-1";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full font-medium",
        config.bg,
        config.text,
        sizeClasses
      )}
    >
      <span className="relative flex h-2 w-2">
        {pulse && status === "running" && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping",
              config.dot
            )}
          />
        )}
        <span
          className={cn("relative inline-flex h-2 w-2 rounded-full", config.dot)}
        />
      </span>
      {displayLabel}
    </span>
  );
}
