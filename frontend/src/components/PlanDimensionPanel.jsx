import React, { useEffect, useMemo, useRef, useState } from 'react'

function PlanDimensionPanel({
  plans,
  dimensionDefs,
  selectedPlanIds,
  selectedDimensions,
  setSelectedPlanIds,
  setSelectedDimensions,
  onRunCompare,
  compareBusy,
}) {
  const [planQuery, setPlanQuery] = useState('')
  const [open, setOpen] = useState(false)
  const boxRef = useRef(null)

  const filteredPlans = useMemo(() => {
    const q = planQuery.trim().toLowerCase()
    if (!q) return plans
    return plans.filter((p) => {
      const fileName = p.source_file.split('\\').pop() || ''
      return p.name.toLowerCase().includes(q) || p.source_file.toLowerCase().includes(q) || fileName.toLowerCase().includes(q)
    })
  }, [plans, planQuery])

  const togglePlan = (planId) => {
    if (selectedPlanIds.includes(planId)) {
      setSelectedPlanIds(selectedPlanIds.filter((id) => id !== planId))
      return
    }
    setSelectedPlanIds([...selectedPlanIds, planId])
  }

  const toggleDimension = (key) => {
    if (selectedDimensions.includes(key)) {
      setSelectedDimensions(selectedDimensions.filter((d) => d !== key))
      return
    }
    setSelectedDimensions([...selectedDimensions, key])
  }

  useEffect(() => {
    const onClickOutside = (event) => {
      if (!boxRef.current) return
      if (!boxRef.current.contains(event.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [])

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold">计划选择</h2>
        <p className="text-xs text-slate-500">从 /data 中解析出的保险计划中选择，至少选择两个进行并列比较</p>
      </div>

      <div className="space-y-2" ref={boxRef}>
        <input
          value={planQuery}
          onChange={(e) => setPlanQuery(e.target.value)}
          onFocus={() => setOpen(true)}
          onClick={() => setOpen(true)}
          placeholder="点击并输入关键词搜索计划"
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm outline-none ring-accent focus:ring"
        />

        {open && (
          <div className="max-h-64 space-y-2 overflow-auto rounded-md border border-slate-200 p-2">
            {plans.length === 0 && (
              <div className="rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-500">
                暂无可选计划。请点击右上角“重新解析数据”。
              </div>
            )}
            {plans.length > 0 && filteredPlans.length === 0 && (
              <div className="rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-500">没有匹配结果，换个关键词试试。</div>
            )}
            {filteredPlans.map((plan) => (
              <label key={plan.plan_id} className="flex cursor-pointer items-start gap-2 rounded-md p-2 hover:bg-slate-50">
                <input
                  type="checkbox"
                  checked={selectedPlanIds.includes(plan.plan_id)}
                  onChange={() => togglePlan(plan.plan_id)}
                  className="mt-1 h-4 w-4 rounded border-slate-300"
                />
                <div>
                  <div className="text-sm font-medium leading-5">{plan.name}</div>
                  <div className="text-xs text-slate-500">{plan.source_file.split('\\').pop()}</div>
                </div>
              </label>
            ))}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          {selectedPlanIds.map((pid) => {
            const plan = plans.find((p) => p.plan_id === pid)
            if (!plan) return null
            return (
              <button
                key={pid}
                type="button"
                onClick={() => togglePlan(pid)}
                className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs text-teal-700"
              >
                {plan.name} ×
              </button>
            )
          })}
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold">维度筛选</h2>
        <div className="mt-2 max-h-72 space-y-1 overflow-auto rounded-md border border-slate-200 p-2">
          {dimensionDefs.map((dim) => (
            <label key={dim.key} className="flex cursor-pointer items-center gap-2 rounded-md p-2 hover:bg-slate-50">
              <input
                type="checkbox"
                checked={selectedDimensions.includes(dim.key)}
                onChange={() => toggleDimension(dim.key)}
                className="h-4 w-4 rounded border-slate-300"
              />
              <span className="text-sm">{dim.label}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        <button
          type="button"
          onClick={onRunCompare}
          disabled={compareBusy || selectedPlanIds.length < 2 || selectedDimensions.length === 0}
          className="w-full rounded-md bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {compareBusy ? '生成中...' : '生成比较表'}
        </button>
        <p className="text-xs text-slate-500">说明: 选择计划和维度后会自动刷新，也可以点击按钮手动触发。</p>
      </div>
    </div>
  )
}

export default PlanDimensionPanel
