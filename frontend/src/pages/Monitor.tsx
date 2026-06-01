import { useEffect, useRef, useState } from "react";
import {
  Activity,
  Cpu,
  Compass,
  Zap,
  TrendingUp,
  TrendingDown,
  ArrowRight,
  Layers,
  BarChart3,
} from "lucide-react";
import { useMonitorStore } from "@/store/monitorStore";
import type { SlotStatus, GeneratorStatus } from "@/store/monitorStore";

const DIRECTIONS = ["momentum", "value", "quality", "volatility", "liquidity", "size"];

interface CurvePoint {
  time: string;
  generated: number;
  passed: number;
}

function slotStatusLabel(status: SlotStatus): string {
  switch (status) {
    case "idle":
      return "空闲";
    case "submitting":
      return "提交中";
    case "waiting":
      return "等待结果";
  }
}

function generatorStatusLabel(status: GeneratorStatus): string {
  switch (status) {
    case "idle":
      return "空闲";
    case "generating":
      return "生成中";
    case "validating":
      return "验证中";
    case "enqueuing":
      return "入队中";
  }
}

function MiniGauge({ value, max }: { value: number; max: number }) {
  const pct = Math.min(value / max, 1);
  const r = 18;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - pct);
  const color = pct > 0.6 ? "#22c55e" : pct > 0.3 ? "#f59e0b" : "#ef4444";
  return (
    <svg width="48" height="48" viewBox="0 0 48 48" className="-rotate-90">
      <circle cx="24" cy="24" r={r} fill="none" stroke="#1f2937" strokeWidth="4" />
      <circle
        cx="24"
        cy="24"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={offset}
        className="transition-all duration-700"
      />
    </svg>
  );
}

function MetricCard({
  label,
  value,
  icon,
  trend,
  colorClass,
  rightElement,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  trend?: "up" | "down" | null;
  colorClass?: string;
  rightElement?: React.ReactNode;
}) {
  return (
    <div className="glass rounded-m3-md flex flex-col gap-m3-2 p-m3-4">
      <div className="flex items-center justify-between">
        <span className="text-body-sm text-m3-on-surface-variant">{label}</span>
        <span className="text-m3-primary">{icon}</span>
      </div>
      <div className="flex items-center gap-m3-2">
        <div className="flex items-end gap-m3-1">
          <span className={`text-title-lg font-medium ${colorClass ?? "text-m3-on-surface"}`}>
            {value}
          </span>
          {trend === "up" && <TrendingUp className="h-4 w-4 text-m3-success" />}
          {trend === "down" && <TrendingDown className="h-4 w-4 text-m3-error" />}
        </div>
        {rightElement && <div className="ml-auto">{rightElement}</div>}
      </div>
    </div>
  );
}

function KeyMetrics({
  totalGenerated,
  totalPassed,
}: {
  totalGenerated: number;
  totalPassed: number;
}) {
  const metrics = useMonitorStore((s) => s.metrics);
  const passRatePct = (metrics.passRate * 100).toFixed(1);
  const sharpeColor =
    metrics.avgSharpe >= 1.25
      ? "text-m3-success"
      : metrics.avgSharpe >= 0.8
        ? "text-m3-primary"
        : "text-m3-error";

  return (
    <div className="grid grid-cols-4 gap-m3-4">
      <MetricCard
        label="总产量"
        value={totalGenerated.toLocaleString()}
        icon={<BarChart3 className="h-4 w-4" />}
        trend={totalGenerated > 0 ? "up" : null}
      />
      <MetricCard
        label="通过数"
        value={totalPassed.toLocaleString()}
        icon={<Zap className="h-4 w-4" />}
        colorClass="text-m3-success"
        trend={totalPassed > 0 ? "up" : null}
      />
      <MetricCard
        label="通过率"
        value={`${passRatePct}%`}
        icon={<Activity className="h-4 w-4" />}
        rightElement={<MiniGauge value={metrics.passRate} max={1} />}
      />
      <MetricCard
        label="平均Sharpe"
        value={metrics.avgSharpe.toFixed(2)}
        icon={<TrendingUp className="h-4 w-4" />}
        colorClass={sharpeColor}
      />
    </div>
  );
}

function ProductionCurve({ curveData }: { curveData: CurvePoint[] }) {
  if (curveData.length < 2) {
    return (
      <div className="glass rounded-m3-md flex flex-col h-full p-m3-4">
        <div className="flex items-center gap-m3-2 mb-m3-4">
          <Activity className="h-4 w-4 text-m3-primary" />
          <h2 className="text-body-md font-medium text-m3-on-surface">产量曲线</h2>
        </div>
        <div className="flex-1 flex items-center justify-center text-body-sm text-m3-on-surface-variant min-h-[200px]">
          等待数据...
        </div>
      </div>
    );
  }

  const w = 520;
  const h = 220;
  const padL = 44;
  const padR = 12;
  const padT = 12;
  const padB = 24;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const maxVal = Math.max(...curveData.map((d) => Math.max(d.generated, d.passed)), 1);
  const niceMax = Math.ceil(maxVal / 5) * 5 || 5;

  const xStep = plotW / (curveData.length - 1);
  const toX = (i: number) => padL + i * xStep;
  const toY = (v: number) => padT + plotH - (v / niceMax) * plotH;

  const genLine = curveData.map((d, i) => `${toX(i)},${toY(d.generated)}`).join(" ");
  const passLine = curveData.map((d, i) => `${toX(i)},${toY(d.passed)}`).join(" ");

  const genArea = [
    ...curveData.map((d, i) => `${toX(i)},${toY(d.generated)}`),
    `${toX(curveData.length - 1)},${padT + plotH}`,
    `${toX(0)},${padT + plotH}`,
  ].join(" ");

  const passArea = [
    ...curveData.map((d, i) => `${toX(i)},${toY(d.passed)}`),
    `${toX(curveData.length - 1)},${padT + plotH}`,
    `${toX(0)},${padT + plotH}`,
  ].join(" ");

  const yTicks = 5;
  const yTickVals = Array.from({ length: yTicks + 1 }, (_, i) =>
    Math.round((niceMax / yTicks) * i)
  );

  const xLabelInterval = Math.max(1, Math.floor(curveData.length / 6));

  return (
    <div className="glass rounded-m3-md flex flex-col p-m3-4">
      <div className="flex items-center gap-m3-2 mb-m3-3">
        <Activity className="h-4 w-4 text-m3-primary" />
        <h2 className="text-body-md font-medium text-m3-on-surface">产量曲线</h2>
        <div className="ml-auto flex items-center gap-m3-4 text-body-sm text-m3-on-surface-variant">
          <span className="flex items-center gap-m3-1">
            <span className="inline-block h-0.5 w-4 rounded bg-amber-500" />
            生成数
          </span>
          <span className="flex items-center gap-m3-1">
            <span className="inline-block h-0.5 w-4 rounded bg-green-500" />
            通过数
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" preserveAspectRatio="xMidYMid meet">
        {yTickVals.map((v) => (
          <g key={v}>
            <line
              x1={padL}
              y1={toY(v)}
              x2={w - padR}
              y2={toY(v)}
              stroke="#1f2937"
              strokeWidth="0.5"
            />
            <text
              x={padL - 6}
              y={toY(v) + 3}
              textAnchor="end"
              fill="#9ca3af"
              fontSize="9"
              fontFamily="JetBrains Mono, monospace"
            >
              {v}
            </text>
          </g>
        ))}
        <polygon points={genArea} fill="rgba(245,158,11,0.06)" />
        <polyline
          points={genLine}
          fill="none"
          stroke="#f59e0b"
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <polygon points={passArea} fill="rgba(34,197,94,0.06)" />
        <polyline
          points={passLine}
          fill="none"
          stroke="#22c55e"
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {curveData.map(
          (d, i) =>
            (i % xLabelInterval === 0 || i === curveData.length - 1) && (
              <text
                key={i}
                x={toX(i)}
                y={h - 4}
                textAnchor="middle"
                fill="#9ca3af"
                fontSize="8"
                fontFamily="JetBrains Mono, monospace"
              >
                {d.time}
              </text>
            )
        )}
      </svg>
    </div>
  );
}

function PipelineStatus() {
  const generators = useMonitorStore((s) => s.generators);
  const slots = useMonitorStore((s) => s.slots);

  const activeGens = generators.filter((g) => g.status !== "idle").length;
  const queueCount = generators.filter((g) => g.status === "enqueuing").length;
  const submittingSlots = slots.filter((s) => s.status !== "idle").length;

  const stages = [
    {
      label: "生成器",
      count: `${activeGens}/${generators.length}`,
      active: activeGens > 0,
      detail: generators.map((g) => generatorStatusLabel(g.status)).join(" / "),
    },
    {
      label: "队列",
      count: String(queueCount),
      active: queueCount > 0,
      detail: queueCount > 0 ? "入队中" : "空闲",
    },
    {
      label: "BRAIN提交",
      count: `${submittingSlots}/${slots.length}`,
      active: submittingSlots > 0,
      detail: slots.map((s) => slotStatusLabel(s.status)).join(" / "),
    },
  ];

  return (
    <div className="glass rounded-m3-md flex flex-col p-m3-4">
      <div className="flex items-center gap-m3-2 mb-m3-4">
        <Layers className="h-4 w-4 text-m3-primary" />
        <h2 className="text-body-md font-medium text-m3-on-surface">流水线状态</h2>
      </div>
      <div className="flex flex-col gap-m3-2">
        {stages.map((stage, i) => (
          <div key={stage.label}>
            <div
              className={`rounded-m3-sm p-m3-3 transition-all ${
                stage.active
                  ? "glass-strong border border-m3-primary/50"
                  : "glass"
              }`}
            >
              <div className="flex items-center justify-between mb-m3-1">
                <span className="text-body-md font-medium text-m3-on-surface">
                  {stage.label}
                </span>
                <div className="flex items-center gap-m3-2">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      stage.active
                        ? "bg-m3-primary animate-pulse"
                        : "bg-m3-on-surface-variant/40"
                    }`}
                  />
                  <span className="font-mono text-body-sm text-m3-on-surface-variant">
                    {stage.count}
                  </span>
                </div>
              </div>
              <span className="text-body-sm text-m3-on-surface-variant">
                {stage.detail}
              </span>
            </div>
            {i < stages.length - 1 && (
              <div className="flex justify-center py-m3-1">
                <ArrowRight className="h-4 w-4 text-m3-on-surface-variant" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function BrainSlots() {
  const slots = useMonitorStore((s) => s.slots);

  return (
    <div className="glass rounded-m3-md flex flex-col p-m3-4">
      <div className="flex items-center gap-m3-2 mb-m3-4">
        <Cpu className="h-4 w-4 text-m3-primary" />
        <h2 className="text-body-md font-medium text-m3-on-surface">BRAIN槽位</h2>
      </div>
      <div className="flex flex-col gap-m3-3">
        {slots.map((slot) => {
          const isActive = slot.status !== "idle";
          return (
            <div
              key={slot.id}
              className={`rounded-m3-sm p-m3-3 transition-all ${
                isActive
                  ? "glass-strong border border-m3-primary/50 animate-pulseOrange"
                  : "glass"
              }`}
            >
              <div className="flex items-center justify-between mb-m3-1">
                <span className="text-body-md font-medium text-m3-on-surface">
                  Slot {slot.id}
                </span>
                <span
                  className={`inline-flex items-center gap-1 rounded-m3-full px-2 py-0.5 text-label-sm ${
                    isActive
                      ? "bg-m3-primary/20 text-m3-primary"
                      : "bg-m3-outline-variant/50 text-m3-on-surface-variant"
                  }`}
                >
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      isActive ? "bg-m3-primary" : "bg-m3-on-surface-variant"
                    }`}
                  />
                  {slotStatusLabel(slot.status)}
                </span>
              </div>
              {slot.alphaExpression ? (
                <p
                  className="truncate font-mono text-body-sm text-m3-on-surface/70"
                  title={slot.alphaExpression}
                >
                  {slot.alphaExpression}
                </p>
              ) : (
                <p className="font-mono text-body-sm text-m3-on-surface-variant/40">—</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MabDirection() {
  const mab = useMonitorStore((s) => s.mab);
  const directionCounts = mab.arms.reduce<Record<string, number>>((acc, arm) => {
    acc[arm.direction] = (acc[arm.direction] ?? 0) + 1;
    return acc;
  }, {});
  const maxCount = Math.max(...Object.values(directionCounts), 1);
  const bestArm = mab.arms.length > 0
    ? mab.arms.reduce((best, cur) => (cur.ucb_score > best.ucb_score ? cur : best), mab.arms[0])
    : null;

  return (
    <div className="glass rounded-m3-md flex flex-col p-m3-4">
      <div className="flex items-center gap-m3-2 mb-m3-4">
        <Compass className="h-4 w-4 text-m3-primary" />
        <h2 className="text-body-md font-medium text-m3-on-surface">MAB方向分布</h2>
      </div>
      <div className="flex flex-col gap-m3-2">
        {DIRECTIONS.map((dir) => {
          const isCurrent = bestArm ? dir === bestArm.direction : false;
          const count = directionCounts[dir] ?? 0;
          const widthPct = (count / maxCount) * 100;
          return (
            <div key={dir} className="flex items-center gap-m3-2">
              <span
                className={`w-20 text-body-sm font-medium text-right shrink-0 ${
                  isCurrent ? "text-m3-primary" : "text-m3-on-surface-variant"
                }`}
              >
                {isCurrent && <Zap className="inline h-3 w-3 mr-0.5 -mt-0.5" />}
                {dir}
              </span>
              <div className="flex-1 h-5 rounded-m3-sm bg-m3-surface-container-lowest overflow-hidden">
                <div
                  className={`h-full rounded-m3-sm transition-all duration-500 ${
                    isCurrent ? "bg-m3-primary animate-pulseOrange" : "bg-m3-outline"
                  }`}
                  style={{ width: `${Math.max(widthPct, 2)}%` }}
                />
              </div>
              <span className="w-8 text-right font-mono text-body-sm text-m3-on-surface-variant shrink-0">
                {count}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AlphaBacklog() {
  const generators = useMonitorStore((s) => s.generators);
  const slots = useMonitorStore((s) => s.slots);

  const pending = generators.filter(
    (g) => g.status === "generating" || g.status === "validating"
  ).length;
  const inQueue = generators.filter((g) => g.status === "enqueuing").length;
  const submitted = slots.filter((s) => s.status === "submitting").length;
  const awaiting = slots.filter((s) => s.status === "waiting").length;

  const items = [
    { label: "待处理", count: pending, color: "text-m3-primary" },
    { label: "队列中", count: inQueue, color: "text-amber-400" },
    { label: "已提交", count: submitted, color: "text-blue-400" },
    { label: "待审核", count: awaiting, color: "text-m3-success" },
  ];

  return (
    <div className="glass rounded-m3-md flex flex-col p-m3-4">
      <div className="flex items-center gap-m3-2 mb-m3-4">
        <Layers className="h-4 w-4 text-m3-primary" />
        <h2 className="text-body-md font-medium text-m3-on-surface">Alpha堆积监控</h2>
      </div>
      <div className="grid grid-cols-2 gap-m3-3">
        {items.map((item) => (
          <div
            key={item.label}
            className="glass rounded-m3-sm p-m3-3 text-center"
          >
            <div className={`text-title-lg font-medium font-mono ${item.color}`}>
              {item.count}
            </div>
            <div className="text-body-sm text-m3-on-surface-variant mt-m3-1">
              {item.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Monitor() {
  const fetchMab = useMonitorStore((s) => s.fetchMab);
  const fetchOverview = useMonitorStore((s) => s.fetchOverview);
  const connectWs = useMonitorStore((s) => s.connectWs);
  const disconnectWs = useMonitorStore((s) => s.disconnectWs);
  const connected = useMonitorStore((s) => s.connected);
  const generators = useMonitorStore((s) => s.generators);
  const metrics = useMonitorStore((s) => s.metrics);

  const [totalGenerated, setTotalGenerated] = useState(0);
  const [totalPassed, setTotalPassed] = useState(0);
  const [curveData, setCurveData] = useState<CurvePoint[]>([]);

  const prevGeneratorsRef = useRef(generators);
  const totalGenRef = useRef(totalGenerated);
  const totalPassRef = useRef(totalPassed);

  totalGenRef.current = totalGenerated;
  totalPassRef.current = totalPassed;

  useEffect(() => {
    const prev = prevGeneratorsRef.current;
    let delta = 0;
    generators.forEach((gen, i) => {
      if (prev[i] && prev[i].status === "enqueuing" && gen.status === "idle") {
        delta++;
      }
    });
    if (delta > 0) {
      setTotalGenerated((v) => v + delta);
    }
    prevGeneratorsRef.current = generators;
  }, [generators]);

  useEffect(() => {
    setTotalPassed(Math.round(metrics.passRate * totalGenRef.current));
  }, [metrics.passRate]);

  useEffect(() => {
    fetchMab();
    fetchOverview();
    connectWs();
    return () => {
      disconnectWs();
    };
  }, [fetchMab, fetchOverview, connectWs, disconnectWs]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchMab();
      fetchOverview();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchMab, fetchOverview]);

  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const timeLabel = `${now
        .getHours()
        .toString()
        .padStart(2, "0")}:${now
        .getMinutes()
        .toString()
        .padStart(2, "0")}:${now
        .getSeconds()
        .toString()
        .padStart(2, "0")}`;
      setCurveData((prev) => {
        const next = [
          ...prev,
          { time: timeLabel, generated: totalGenRef.current, passed: totalPassRef.current },
        ];
        return next.slice(-20);
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-full flex-col gap-6">
      <div className="flex items-center gap-m3-3">
        <Activity className="h-7 w-7 text-m3-primary" />
        <h1 className="text-headline-sm font-medium text-m3-on-surface">实时监控</h1>
        <div className="ml-auto flex items-center gap-m3-2">
          <div
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-m3-success animate-pulse" : "bg-m3-error"
            }`}
          />
          <span className="text-body-sm text-m3-on-surface-variant">
            {connected ? "已连接" : "已断开"}
          </span>
        </div>
      </div>

      <KeyMetrics totalGenerated={totalGenerated} totalPassed={totalPassed} />

      <div className="grid grid-cols-3 gap-m3-4">
        <div className="col-span-2">
          <ProductionCurve curveData={curveData} />
        </div>
        <PipelineStatus />
      </div>

      <div className="grid grid-cols-3 gap-m3-4">
        <BrainSlots />
        <MabDirection />
        <AlphaBacklog />
      </div>
    </div>
  );
}
