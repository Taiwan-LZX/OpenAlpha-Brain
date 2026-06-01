import { useState, useEffect, useMemo } from 'react'
import { Target, Grid3x3, UserCheck, AlertTriangle, Loader2 } from 'lucide-react'
import { useAlgoStore } from '@/store/algoStore'
import { cn } from '@/lib/utils'

type Tab = 'mab' | 'featureMap' | 'strategy' | 'decay'

const TABS: { key: Tab; label: string; icon: typeof Target }[] = [
  { key: 'mab', label: 'MAB多臂老虎机', icon: Target },
  { key: 'featureMap', label: '特征图谱', icon: Grid3x3 },
  { key: 'strategy', label: '策略画像', icon: UserCheck },
  { key: 'decay', label: '衰减检测', icon: AlertTriangle },
]

const DIRECTIONS = ['momentum', 'value', 'quality', 'volatility', 'liquidity', 'size']
const TIMESCALES = ['short', 'mid', 'long']
const MECHANISMS = ['cross_section', 'time_series', 'interaction', 'nonlinear']

function ucbToColor(score: number, maxScore: number): string {
  if (maxScore === 0) return '#1a1a2e'
  const ratio = Math.min(score / maxScore, 1)
  const r = Math.round(26 + (245 - 26) * ratio)
  const g = Math.round(26 + (158 - 26) * ratio)
  const b = Math.round(46 + (11 - 46) * ratio)
  return `rgb(${r}, ${g}, ${b})`
}

function eliteToColor(count: number, maxCount: number): string {
  if (maxCount === 0) return '#111122'
  if (count === 0) return '#111122'
  const ratio = Math.min(count / maxCount, 1)
  const r = Math.round(26 + (245 - 26) * ratio)
  const g = Math.round(26 + (158 - 26) * ratio)
  const b = Math.round(46 + (11 - 46) * ratio)
  return `rgb(${r}, ${g}, ${b})`
}

function StatCard({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="flex flex-col gap-m3-1 rounded-m3-sm bg-m3-surface-container-high px-m3-4 py-m3-3 border border-m3-outline-variant">
      <span className="text-label-sm text-m3-on-surface-variant">{label}</span>
      <span className={cn('text-title-md font-mono', accent ? 'text-m3-primary' : 'text-m3-on-surface')}>
        {value}
      </span>
    </div>
  )
}

export default function Algorithm() {
  const [activeTab, setActiveTab] = useState<Tab>('mab')
  const mab = useAlgoStore((s) => s.mab)
  const featureMap = useAlgoStore((s) => s.featureMap)
  const strategyProfiles = useAlgoStore((s) => s.strategyProfiles)
  const decayBlacklist = useAlgoStore((s) => s.decayBlacklist)
  const decayRecords = useAlgoStore((s) => s.decayRecords)
  const loading = useAlgoStore((s) => s.loading)
  const error = useAlgoStore((s) => s.error)
  const fetchAll = useAlgoStore((s) => s.fetchAll)

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

  const mabDirections = useMemo(() => [...new Set(mab.map((a) => a.direction))], [mab])
  const maxUcb = useMemo(() => Math.max(...mab.map((a) => a.ucb_score), 0), [mab])

  const mabMap = useMemo(() => {
    const map = new Map<string, { ucb_score: number; expectation: number }>()
    mab.forEach((a) => map.set(a.direction, { ucb_score: a.ucb_score, expectation: a.expectation }))
    return map
  }, [mab])

  const bestArm = useMemo(() => {
    if (mab.length === 0) return null
    return mab.reduce((best, cur) => (cur.ucb_score > best.ucb_score ? cur : best), mab[0])
  }, [mab])

  const fmMaxElite = useMemo(() => Math.max(...featureMap.map((c) => c.elite_count), 0), [featureMap])

  const fmMap = useMemo(() => {
    const map = new Map<string, number>()
    featureMap.forEach((c) => map.set(`${c.direction}-${c.time_horizon}-${c.mechanism}`, c.elite_count))
    return map
  }, [featureMap])

  const coverage = useMemo(() => {
    const total = DIRECTIONS.length * TIMESCALES.length * MECHANISMS.length
    let filled = 0
    DIRECTIONS.forEach((d) => {
      TIMESCALES.forEach((ts) => {
        MECHANISMS.forEach((m) => {
          const count = fmMap.get(`${d}-${ts}-${m}`) ?? 0
          if (count > 0) filled++
        })
      })
    })
    return Math.round((filled / total) * 100)
  }, [fmMap])

  const totalElites = useMemo(() => featureMap.reduce((sum, c) => sum + c.elite_count, 0), [featureMap])

  const directionDistribution = useMemo(() => {
    const dist: Record<string, number> = {}
    DIRECTIONS.forEach((d) => (dist[d] = 0))
    strategyProfiles.forEach((p) => {
      dist[p.direction] = (dist[p.direction] ?? 0) + 1
    })
    return dist
  }, [strategyProfiles])

  const maxDistCount = useMemo(
    () => Math.max(...Object.values(directionDistribution), 1),
    [directionDistribution]
  )

  const bestSharpeProfile = useMemo(() => {
    if (strategyProfiles.length === 0) return null
    return strategyProfiles.reduce((best, cur) => (cur.sharpe > best.sharpe ? cur : best), strategyProfiles[0])
  }, [strategyProfiles])

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-m3-3">
        <Target className="h-6 w-6 text-m3-primary" />
        <h1 className="text-headline-sm font-medium text-white">算法内部状态</h1>
      </div>

        {error && (
          <div className="rounded-m3-sm bg-m3-error/10 border border-m3-error/30 px-m3-4 py-m3-3 text-body-md text-m3-error">
            {error}
          </div>
        )}

        <div className="flex gap-1 rounded-m3-sm bg-m3-surface-container p-1 border border-m3-outline-variant">
          {TABS.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={cn(
                  'flex items-center gap-m3-2 rounded-md px-m3-4 py-2 text-body-md font-medium transition-colors',
                  activeTab === tab.key
                    ? 'bg-m3-primary text-m3-on-primary'
                    : 'text-m3-on-surface-variant hover:text-m3-on-surface'
                )}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-8 w-8 animate-spin text-m3-primary" />
          </div>
        ) : (
          <>
            {activeTab === 'mab' && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-m3-3 sm:grid-cols-4">
                  <StatCard label="总臂数" value={mab.length} />
                  <StatCard label="最佳方向" value={bestArm ? bestArm.direction : '-'} accent />
                  <StatCard label="最高UCB" value={bestArm ? bestArm.ucb_score.toFixed(3) : '-'} accent />
                  <StatCard label="最高期望" value={bestArm ? bestArm.expectation.toFixed(3) : '-'} accent />
                </div>

                <div className="glass rounded-m3-md p-m3-4">
                  <h3 className="text-body-md font-medium text-m3-on-surface-variant mb-4">UCB分数</h3>
                  {mab.length === 0 ? (
                    <div className="py-12 text-center text-m3-on-surface-variant">暂无数据</div>
                  ) : (
                    <div className="space-y-2">
                      {[...mab]
                        .sort((a, b) => b.ucb_score - a.ucb_score)
                        .map((arm) => (
                          <div key={arm.direction} className="flex items-center gap-m3-3">
                            <div className="w-36 text-body-sm text-m3-on-surface-variant truncate">
                              {arm.direction}
                            </div>
                            <div className="flex-1 h-6 bg-m3-surface-container-lowest rounded overflow-hidden">
                              <div
                                className="h-full rounded transition-all"
                                style={{
                                  width: `${maxUcb > 0 ? (arm.ucb_score / maxUcb) * 100 : 0}%`,
                                  backgroundColor: '#f59e0b',
                                }}
                              />
                            </div>
                            <div className="w-20 text-right text-body-sm font-mono text-white">
                              {arm.ucb_score.toFixed(2)}
                            </div>
                          </div>
                        ))}
                    </div>
                  )}
                </div>

                <div className="glass rounded-m3-md p-m3-4">
                  <h3 className="text-body-md font-medium text-m3-on-surface-variant mb-4">臂详情</h3>
                  {mab.length === 0 ? (
                    <div className="py-12 text-center text-m3-on-surface-variant">暂无数据</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-body-md">
                        <thead>
                          <tr className="border-b border-m3-outline-variant text-m3-on-surface-variant">
                            <th className="px-m3-4 py-m3-2 text-left font-medium">方向</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">Alpha</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">Beta</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">期望</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">UCB</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...mab]
                            .sort((a, b) => b.ucb_score - a.ucb_score)
                            .map((arm, i) => (
                              <tr key={i} className="border-b border-m3-outline-variant/50">
                                <td className="px-m3-4 py-m3-2 text-white">{arm.direction}</td>
                                <td className="px-m3-4 py-m3-2 font-mono text-white">{arm.alpha.toFixed(2)}</td>
                                <td className="px-m3-4 py-m3-2 font-mono text-white">{arm.beta.toFixed(2)}</td>
                                <td className="px-m3-4 py-m3-2 font-mono text-white">{arm.expectation.toFixed(3)}</td>
                                <td className="px-m3-4 py-m3-2 font-mono text-m3-primary font-medium">{arm.ucb_score.toFixed(3)}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'featureMap' && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-m3-3 sm:grid-cols-4">
                  <StatCard label="覆盖率" value={`${coverage}%`} accent />
                  <StatCard label="总Elite数" value={totalElites} />
                  <StatCard label="已覆盖单元" value={featureMap.filter((c) => c.elite_count > 0).length} />
                  <StatCard label="总单元数" value={DIRECTIONS.length * TIMESCALES.length * MECHANISMS.length} />
                </div>

                <div className="glass rounded-m3-md p-m3-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-body-md font-medium text-m3-on-surface-variant">覆盖率</h3>
                    <span className="text-body-md font-mono text-m3-primary">{coverage}%</span>
                  </div>
                  <div className="h-3 bg-m3-surface-container-lowest rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{ width: `${coverage}%`, backgroundColor: '#f59e0b' }}
                    />
                  </div>
                </div>

                <div className="glass rounded-m3-md p-m3-4">
                  <h3 className="text-body-md font-medium text-m3-on-surface-variant mb-2">特征图谱网格</h3>
                  <div className="flex gap-m3-4 mb-4 text-body-sm text-m3-on-surface-variant">
                    <span>机制:</span>
                    {MECHANISMS.map((m, i) => (
                      <span key={m} className="text-m3-on-surface-variant/70">
                        {i + 1}={m}
                      </span>
                    ))}
                  </div>
                  {featureMap.length === 0 && fmMap.size === 0 ? (
                    <div className="py-12 text-center text-m3-on-surface-variant">暂无数据</div>
                  ) : (
                    <div className="space-y-5">
                      {DIRECTIONS.map((dir) => (
                        <div key={dir}>
                          <div className="text-body-sm text-m3-on-surface-variant font-medium mb-2 capitalize">{dir}</div>
                          <div className="space-y-1">
                            {TIMESCALES.map((ts) => (
                              <div key={ts} className="flex items-center gap-1">
                                <span className="w-10 text-body-sm text-m3-on-surface-variant/70">{ts}</span>
                                <div className="flex gap-1">
                                  {MECHANISMS.map((mech) => {
                                    const count = fmMap.get(`${dir}-${ts}-${mech}`) ?? 0
                                    return (
                                      <div
                                        key={mech}
                                        className="w-[4.5rem] h-10 rounded-m3-xs flex items-center justify-center text-body-sm font-mono cursor-default transition-transform hover:scale-105"
                                        style={{
                                          backgroundColor: eliteToColor(count, fmMaxElite),
                                          color: count > 0 ? '#fff' : '#444',
                                        }}
                                        title={`方向: ${dir}\n时间尺度: ${ts}\n机制: ${mech}\nElite数: ${count}`}
                                      >
                                        {count > 0 ? count : ''}
                                      </div>
                                    )
                                  })}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'strategy' && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-m3-3 sm:grid-cols-4">
                  <StatCard label="总画像数" value={strategyProfiles.length} />
                  <StatCard label="涉及方向" value={new Set(strategyProfiles.map((p) => p.direction)).size} />
                  <StatCard label="最佳Sharpe" value={bestSharpeProfile ? bestSharpeProfile.sharpe.toFixed(2) : '-'} accent />
                  <StatCard label="最佳方向" value={bestSharpeProfile ? bestSharpeProfile.direction : '-'} accent />
                </div>

                <div className="glass rounded-m3-md p-m3-4">
                  <h3 className="text-body-md font-medium text-m3-on-surface-variant mb-4">方向分布</h3>
                  <div className="space-y-3">
                    {DIRECTIONS.map((dir) => (
                      <div key={dir} className="flex items-center gap-m3-3">
                        <span className="w-24 text-body-sm text-m3-on-surface-variant">{dir}</span>
                        <div className="flex-1 h-6 bg-m3-surface-container-lowest rounded overflow-hidden">
                          <div
                            className="h-full rounded transition-all"
                            style={{
                              width: `${((directionDistribution[dir] ?? 0) / maxDistCount) * 100}%`,
                              backgroundColor: '#f59e0b',
                            }}
                          />
                        </div>
                        <span className="w-8 text-right text-body-sm font-mono text-white">
                          {directionDistribution[dir] ?? 0}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass rounded-m3-md p-m3-4">
                  <h3 className="text-body-md font-medium text-m3-on-surface-variant mb-4">Top策略画像</h3>
                  {strategyProfiles.length === 0 ? (
                    <div className="py-12 text-center text-m3-on-surface-variant">暂无数据</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-body-md">
                        <thead>
                          <tr className="border-b border-m3-outline-variant text-m3-on-surface-variant">
                            <th className="px-m3-4 py-m3-2 text-left font-medium">方向</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">机制</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">时间尺度</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">平均Sharpe</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...strategyProfiles]
                            .sort((a, b) => b.sharpe - a.sharpe)
                            .map((p, i) => (
                              <tr key={i} className="border-b border-m3-outline-variant/50">
                                <td className="px-m3-4 py-m3-2 text-white">{p.direction}</td>
                                <td className="px-m3-4 py-m3-2 text-white">{p.mechanism}</td>
                                <td className="px-m3-4 py-m3-2 text-white">{p.time_horizon}</td>
                                <td className={cn('px-m3-4 py-m3-2 font-mono', p.sharpe >= 1.25 ? 'text-m3-success font-medium' : 'text-white')}>
                                  {p.sharpe.toFixed(2)}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'decay' && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-m3-3 sm:grid-cols-4">
                  <StatCard label="黑名单方向数" value={decayBlacklist.length} accent={decayBlacklist.length > 0} />
                  <StatCard label="衰减记录数" value={decayRecords.length} accent={decayRecords.length > 0} />
                  <StatCard label="总方向数" value={DIRECTIONS.length} />
                  <StatCard label="健康方向" value={DIRECTIONS.length - decayBlacklist.length} />
                </div>

                <div className="glass rounded-m3-md p-m3-4">
                  <h3 className="text-body-md font-medium text-m3-on-surface-variant mb-4">黑名单方向</h3>
                  {decayBlacklist.length === 0 ? (
                    <div className="py-8 text-center text-m3-on-surface-variant">暂无黑名单方向</div>
                  ) : (
                    <div className="flex flex-wrap gap-m3-2">
                      {decayBlacklist.map((dir) => (
                        <span key={dir} className="rounded-m3-full bg-m3-error/20 text-m3-error px-m3-3 py-1 text-body-md font-medium">
                          {dir}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="glass rounded-m3-md p-m3-4">
                  <h3 className="text-body-md font-medium text-m3-on-surface-variant mb-4">衰减记录</h3>
                  {decayRecords.length === 0 ? (
                    <div className="py-12 text-center text-m3-on-surface-variant">暂无衰减记录</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-body-md">
                        <thead>
                          <tr className="border-b border-m3-outline-variant text-m3-on-surface-variant">
                            <th className="px-m3-4 py-m3-2 text-left font-medium">Alpha ID</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">方向</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">衰减等级</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">EWMA Sharpe</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">综合分数</th>
                            <th className="px-m3-4 py-m3-2 text-left font-medium">GARCH异常</th>
                          </tr>
                        </thead>
                        <tbody>
                          {decayRecords.map((a, i) => (
                            <tr key={i} className="border-b border-m3-outline-variant/50">
                              <td className="px-m3-4 py-m3-2 font-mono text-body-sm text-m3-on-surface max-w-xs truncate" title={a.alpha_id}>
                                {a.alpha_id}
                              </td>
                              <td className="px-m3-4 py-m3-2 text-white">{a.direction}</td>
                              <td className="px-m3-4 py-m3-2">
                                <span className="rounded-m3-full bg-m3-error/20 text-m3-error px-2 py-0.5 text-body-sm font-medium">
                                  {a.decay_level}
                                </span>
                              </td>
                              <td className="px-m3-4 py-m3-2 font-mono text-white">{a.ewma_sharpe.toFixed(3)}</td>
                              <td className="px-m3-4 py-m3-2 font-mono text-white">{a.composite_score.toFixed(3)}</td>
                              <td className="px-m3-4 py-m3-2 text-body-sm">
                                {a.garch_anomaly ? (
                                  <span className="text-m3-error">是</span>
                                ) : (
                                  <span className="text-m3-on-surface-variant">否</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
  )
}
