import { useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { createChart, ColorType } from 'lightweight-charts'
import { ArrowLeft, TrendingUp, Activity, RotateCw, Clock, Loader2 } from 'lucide-react'
import { useAlphaStore, type AlphaStatus } from '@/store/alphaStore'
import { cn } from '@/lib/utils'

const STATUS_STYLES: Record<AlphaStatus, string> = {
  pass: 'bg-m3-success/20 text-m3-success',
  fail: 'bg-m3-error/20 text-m3-error',
  pending: 'bg-gray-500/20 text-gray-400',
}

export default function AlphaDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const currentAlpha = useAlphaStore((s) => s.currentAlpha)
  const loading = useAlphaStore((s) => s.loading)
  const error = useAlphaStore((s) => s.error)
  const fetchAlphaDetail = useAlphaStore((s) => s.fetchAlphaDetail)

  useEffect(() => {
    if (id) fetchAlphaDetail(id)
  }, [id, fetchAlphaDetail])

  useEffect(() => {
    if (!chartContainerRef.current || !currentAlpha?.pnlData?.length) return

    const container = chartContainerRef.current
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: '#1a1a2e' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#2a2a3e' },
        horzLines: { color: '#2a2a3e' },
      },
      width: container.clientWidth,
      height: 400,
      timeScale: {
        timeVisible: true,
      },
    })

    const lineSeries = chart.addLineSeries({
      color: '#f59e0b',
      lineWidth: 2,
    })

    lineSeries.setData(currentAlpha.pnlData)
    chart.timeScale().fitContent()

    const resizeObserver = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth })
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
    }
  }, [currentAlpha])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-m3-primary" />
      </div>
    )
  }

  if (!currentAlpha) {
    return (
      <div className="flex flex-col items-center justify-center gap-m3-4 py-20">
        <p className="text-m3-on-surface-variant">Alpha 未找到</p>
        <button
          onClick={() => navigate('/alphas')}
          className="text-body-md text-m3-primary hover:underline"
        >
          返回列表
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
        <div className="flex items-center gap-m3-4">
          <button
            onClick={() => navigate('/alphas')}
            className="flex items-center gap-1 text-m3-on-surface-variant hover:text-white transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            返回
          </button>
          <h1 className="text-title-lg font-mono text-white">{currentAlpha.id}</h1>
          <span className={cn('rounded-full px-3 py-1 text-body-sm font-medium uppercase', STATUS_STYLES[currentAlpha.status])}>
            {currentAlpha.status}
          </span>
        </div>

        {error && (
          <div className="rounded-m3-sm bg-m3-error/10 border border-m3-error/30 px-m3-4 py-m3-3 text-body-md text-m3-error">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-m3-4">
          <div className="glass rounded-m3-md p-m3-4 space-y-5">
            <h2 className="text-body-md font-medium text-m3-on-surface-variant">基本信息</h2>

            <div className="space-y-4">
              <div>
                <div className="text-body-sm text-m3-on-surface-variant mb-1">Expression</div>
                <div className="font-mono text-body-md text-gray-200 break-all">{currentAlpha.expression}</div>
              </div>

              <div className="grid grid-cols-2 gap-m3-4">
                <div>
                  <div className="flex items-center gap-1 text-body-sm text-m3-on-surface-variant mb-1">
                    <TrendingUp className="h-3 w-3" />
                    Direction
                  </div>
                  <div className="text-body-md text-white capitalize">{currentAlpha.direction}</div>
                </div>
                <div>
                  <div className="flex items-center gap-1 text-body-sm text-m3-on-surface-variant mb-1">
                    <Activity className="h-3 w-3" />
                    Sharpe
                  </div>
                  <div className={cn('text-body-md font-mono', currentAlpha.sharpe >= 1.25 ? 'text-m3-success font-medium' : 'text-white')}>
                    {currentAlpha.sharpe.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-1 text-body-sm text-m3-on-surface-variant mb-1">
                    <Activity className="h-3 w-3" />
                    Fitness
                  </div>
                  <div className={cn('text-body-md font-mono', currentAlpha.fitness > 1 ? 'text-m3-success' : 'text-white')}>
                    {currentAlpha.fitness.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-1 text-body-sm text-m3-on-surface-variant mb-1">
                    <RotateCw className="h-3 w-3" />
                    Turnover
                  </div>
                  <div className="text-body-md font-mono text-white">{currentAlpha.turnover.toFixed(2)}</div>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-1 text-body-sm text-m3-on-surface-variant mb-1">
                  <Clock className="h-3 w-3" />
                  Submitted At
                </div>
                <div className="text-body-md text-white">{new Date(currentAlpha.submittedAt).toLocaleString()}</div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 glass rounded-m3-md p-m3-4">
            <h2 className="text-body-md font-medium text-m3-on-surface-variant mb-4">PnL 曲线</h2>
            {currentAlpha.pnlData?.length ? (
              <div ref={chartContainerRef} />
            ) : (
              <div className="flex items-center justify-center h-[400px] text-m3-on-surface-variant">
                暂无数据
              </div>
            )}
          </div>
      </div>
    </div>
  )
}
