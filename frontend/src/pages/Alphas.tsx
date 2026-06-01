import { useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Filter, TrendingUp, CheckCircle, BarChart3, ChevronRight, Loader2 } from 'lucide-react'
import { useAlphaStore, type Direction, type AlphaStatus } from '@/store/alphaStore'
import { cn } from '@/lib/utils'

const DIRECTIONS: { value: Direction | 'all'; label: string }[] = [
  { value: 'all', label: '全部' },
  { value: 'momentum', label: 'Momentum' },
  { value: 'value', label: 'Value' },
  { value: 'quality', label: 'Quality' },
  { value: 'volatility', label: 'Volatility' },
  { value: 'liquidity', label: 'Liquidity' },
  { value: 'size', label: 'Size' },
]

const STATUSES: { value: AlphaStatus | 'all'; label: string }[] = [
  { value: 'all', label: '全部' },
  { value: 'pass', label: '通过' },
  { value: 'fail', label: '失败' },
  { value: 'pending', label: '待审核' },
]

const DIRECTION_COLORS: Record<Direction, string> = {
  momentum: 'bg-blue-500/20 text-blue-400',
  value: 'bg-green-500/20 text-green-400',
  quality: 'bg-purple-500/20 text-purple-400',
  volatility: 'bg-red-500/20 text-red-400',
  liquidity: 'bg-cyan-500/20 text-cyan-400',
  size: 'bg-amber-500/20 text-amber-400',
}

const STATUS_STYLES: Record<AlphaStatus, string> = {
  pass: 'bg-m3-success/20 text-m3-success',
  fail: 'bg-m3-error/20 text-m3-error',
  pending: 'bg-gray-500/20 text-gray-400',
}

function truncate(str: string, len: number) {
  return str.length > len ? str.slice(0, len) + '...' : str
}

export default function Alphas() {
  const navigate = useNavigate()
  const alphas = useAlphaStore((s) => s.alphas)
  const filters = useAlphaStore((s) => s.filters)
  const loading = useAlphaStore((s) => s.loading)
  const error = useAlphaStore((s) => s.error)
  const setFilters = useAlphaStore((s) => s.setFilters)
  const fetchAlphas = useAlphaStore((s) => s.fetchAlphas)

  useEffect(() => {
    fetchAlphas()
  }, [fetchAlphas])

  const filteredAlphas = useMemo(() => {
    return alphas.filter((a) => {
      if (filters.direction !== 'all' && a.direction !== filters.direction) return false
      if (filters.status !== 'all' && a.status !== filters.status) return false
      if (filters.sharpeMin && a.sharpe < Number(filters.sharpeMin)) return false
      if (filters.sharpeMax && a.sharpe > Number(filters.sharpeMax)) return false
      return true
    })
  }, [alphas, filters])

  const stats = useMemo(() => {
    const total = filteredAlphas.length
    const passed = filteredAlphas.filter((a) => a.status === 'pass').length
    const avgSharpe = total > 0 ? filteredAlphas.reduce((sum, a) => sum + a.sharpe, 0) / total : 0
    return { total, passed, avgSharpe }
  }, [filteredAlphas])

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-m3-3">
        <TrendingUp className="h-6 w-6 text-m3-primary" />
        <h1 className="text-headline-sm font-medium text-white">Alpha 矿场</h1>
      </div>

        {error && (
          <div className="rounded-m3-sm bg-m3-error/10 border border-m3-error/30 px-m3-4 py-m3-3 text-body-md text-m3-error">
            {error}
          </div>
        )}

        <div className="glass rounded-m3-md p-m3-4">
          <div className="flex items-center gap-m3-2 mb-3 text-m3-on-surface-variant">
            <Filter className="h-4 w-4" />
            <span className="text-body-md font-medium">筛选</span>
          </div>
          <div className="flex flex-wrap items-center gap-m3-4">
            <div className="flex items-center gap-m3-2">
              <label className="text-body-md text-m3-on-surface-variant">方向</label>
              <select
                value={filters.direction}
                onChange={(e) => setFilters({ direction: e.target.value as Direction | 'all' })}
                className="rounded-md bg-m3-surface-container-lowest border border-m3-outline-variant px-m3-3 py-m3-2 text-body-md text-white focus:outline-none focus:ring-1 focus:ring-m3-primary"
              >
                {DIRECTIONS.map((d) => (
                  <option key={d.value} value={d.value}>{d.label}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-m3-2">
              <label className="text-body-md text-m3-on-surface-variant">状态</label>
              <select
                value={filters.status}
                onChange={(e) => setFilters({ status: e.target.value as AlphaStatus | 'all' })}
                className="rounded-md bg-m3-surface-container-lowest border border-m3-outline-variant px-m3-3 py-m3-2 text-body-md text-white focus:outline-none focus:ring-1 focus:ring-m3-primary"
              >
                {STATUSES.map((s) => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-m3-2">
              <label className="text-body-md text-m3-on-surface-variant">Sharpe</label>
              <input
                type="number"
                placeholder="最小"
                value={filters.sharpeMin}
                onChange={(e) => setFilters({ sharpeMin: e.target.value })}
                className="w-20 rounded-md bg-m3-surface-container-lowest border border-m3-outline-variant px-2 py-m3-2 text-body-md text-white placeholder:text-m3-on-surface-variant/50 focus:outline-none focus:ring-1 focus:ring-m3-primary"
              />
              <span className="text-m3-on-surface-variant">-</span>
              <input
                type="number"
                placeholder="最大"
                value={filters.sharpeMax}
                onChange={(e) => setFilters({ sharpeMax: e.target.value })}
                className="w-20 rounded-md bg-m3-surface-container-lowest border border-m3-outline-variant px-2 py-m3-2 text-body-md text-white placeholder:text-m3-on-surface-variant/50 focus:outline-none focus:ring-1 focus:ring-m3-primary"
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-m3-4">
          <div className="rounded-m3-sm bg-m3-surface-container p-m3-4 border border-m3-outline-variant">
            <div className="flex items-center gap-m3-2 text-m3-on-surface-variant text-body-md">
              <BarChart3 className="h-4 w-4" />
              总Alpha数
            </div>
            <div className="mt-2 text-headline-sm font-medium text-white">{stats.total}</div>
          </div>
          <div className="rounded-m3-sm bg-m3-surface-container p-m3-4 border border-m3-outline-variant">
            <div className="flex items-center gap-m3-2 text-m3-on-surface-variant text-body-md">
              <CheckCircle className="h-4 w-4" />
              通过数
            </div>
            <div className="mt-2 text-headline-sm font-medium text-m3-success">{stats.passed}</div>
          </div>
          <div className="rounded-m3-sm bg-m3-surface-container p-m3-4 border border-m3-outline-variant">
            <div className="flex items-center gap-m3-2 text-m3-on-surface-variant text-body-md">
              <TrendingUp className="h-4 w-4" />
              平均Sharpe
            </div>
            <div className="mt-2 text-headline-sm font-medium text-m3-primary">{(stats.avgSharpe ?? 0).toFixed(2)}</div>
          </div>
        </div>

        <div className="glass rounded-m3-md overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="h-8 w-8 animate-spin text-m3-primary" />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-body-md">
                <thead>
                  <tr className="border-b border-m3-outline-variant text-m3-on-surface-variant">
                    <th className="px-m3-4 py-m3-3 text-left font-medium">编号</th>
                    <th className="px-m3-4 py-m3-3 text-left font-medium">表达式</th>
                    <th className="px-m3-4 py-m3-3 text-left font-medium">方向</th>
                    <th className="px-m3-4 py-m3-3 text-left font-medium">Sharpe</th>
                    <th className="px-m3-4 py-m3-3 text-left font-medium">Fitness</th>
                    <th className="px-m3-4 py-m3-3 text-left font-medium">Turnover</th>
                    <th className="px-m3-4 py-m3-3 text-left font-medium">状态</th>
                    <th className="px-m3-4 py-m3-3 text-left font-medium">提交时间</th>
                    <th className="px-m3-4 py-m3-3 text-left font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAlphas.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="px-m3-4 py-12 text-center text-m3-on-surface-variant">
                        暂无数据
                      </td>
                    </tr>
                  ) : (
                    filteredAlphas.map((alpha, idx) => (
                      <tr
                        key={`${alpha.id}-${idx}`}
                        onClick={() => navigate(`/alphas/${alpha.id}`)}
                        className="border-b border-m3-outline-variant/50 cursor-pointer transition-colors hover:bg-m3-outline-variant/30"
                      >
                        <td className="px-m3-4 py-m3-3 font-mono text-body-sm text-m3-on-surface-variant">
                          {truncate(alpha.id, 8)}
                        </td>
                        <td className="px-m3-4 py-m3-3" title={alpha.expression}>
                          <span className="font-mono text-body-sm text-gray-300">
                            {truncate(alpha.expression, 30)}
                          </span>
                        </td>
                        <td className="px-m3-4 py-m3-3">
                          <span className={cn('inline-block rounded-full px-2 py-0.5 text-body-sm font-medium', DIRECTION_COLORS[alpha.direction])}>
                            {alpha.direction}
                          </span>
                        </td>
                        <td className={cn('px-m3-4 py-m3-3 font-mono', (alpha.sharpe ?? 0) >= 1.25 ? 'text-m3-success font-medium' : 'text-white')}>
                          {(alpha.sharpe ?? 0).toFixed(2)}
                        </td>
                        <td className={cn('px-m3-4 py-m3-3 font-mono', (alpha.fitness ?? 0) > 1 ? 'text-m3-success' : 'text-white')}>
                          {(alpha.fitness ?? 0).toFixed(2)}
                        </td>
                        <td className="px-m3-4 py-m3-3 font-mono text-white">
                          {(alpha.turnover ?? 0).toFixed(2)}
                        </td>
                        <td className="px-m3-4 py-m3-3">
                          <span className={cn('inline-block rounded-full px-2 py-0.5 text-body-sm font-medium uppercase', STATUS_STYLES[alpha.status])}>
                            {alpha.status}
                          </span>
                        </td>
                        <td className="px-m3-4 py-m3-3 text-m3-on-surface-variant text-body-sm">
                          {new Date(alpha.submittedAt).toLocaleString()}
                        </td>
                        <td className="px-m3-4 py-m3-3">
                          <ChevronRight className="h-4 w-4 text-m3-on-surface-variant" />
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
      </div>
    </div>
  )
}
