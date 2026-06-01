import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

type TrendDirection = "up" | "down" | "flat";

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  trend?: TrendDirection;
  trendValue?: string;
  pulse?: boolean;
  className?: string;
}

function TrendIcon({ direction }: { direction: TrendDirection }) {
  if (direction === "up")
    return <TrendingUp className="h-3.5 w-3.5 text-mine-success" />;
  if (direction === "down")
    return <TrendingDown className="h-3.5 w-3.5 text-mine-fail" />;
  return <Minus className="h-3.5 w-3.5 text-mine-muted" />;
}

export default function MetricCard({
  title,
  value,
  unit,
  trend,
  trendValue,
  pulse = false,
  className,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border border-mine-border bg-mine-card p-4 transition-colors",
        pulse && "animate-glow",
        className
      )}
    >
      {pulse && (
        <div className="absolute inset-0 animate-shimmer pointer-events-none" />
      )}

      <div className="relative">
        <p className="text-xs font-medium text-mine-muted uppercase tracking-wider mb-1">
          {title}
        </p>

        <div className="flex items-baseline gap-1.5">
          <span className="font-mono-title text-2xl font-semibold text-mine-warmWhite">
            {value}
          </span>
          {unit && (
            <span className="text-sm text-mine-muted">{unit}</span>
          )}
        </div>

        {(trend || trendValue) && (
          <div className="flex items-center gap-1 mt-1.5">
            {trend && <TrendIcon direction={trend} />}
            {trendValue && (
              <span
                className={cn(
                  "text-xs font-medium",
                  trend === "up" && "text-mine-success",
                  trend === "down" && "text-mine-fail",
                  trend === "flat" && "text-mine-muted",
                  !trend && "text-mine-muted"
                )}
              >
                {trendValue}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
