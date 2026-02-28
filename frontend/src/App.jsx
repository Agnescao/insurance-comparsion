import React, { useEffect, useMemo, useState } from 'react'

import { createSession, fetchDimensions, fetchPlans, postChatMessage, runCompare, runIngestion } from './api'
import ChatPanel from './components/ChatPanel'
import CompareTable from './components/CompareTable'
import PlanDimensionPanel from './components/PlanDimensionPanel'

function App() {
  const [plans, setPlans] = useState([])
  const [dimensionDefs, setDimensionDefs] = useState([])
  const [selectedPlanIds, setSelectedPlanIds] = useState([])
  const [selectedDimensions, setSelectedDimensions] = useState([])
  const [compareData, setCompareData] = useState(null)
  const [sessionId, setSessionId] = useState('')
  const [chatTurns, setChatTurns] = useState([])
  const [busy, setBusy] = useState(false)
  const [compareBusy, setCompareBusy] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [toast, setToast] = useState('')

  const planMap = useMemo(() => {
    const mapping = {}
    plans.forEach((p) => {
      mapping[p.plan_id] = p
    })
    return mapping
  }, [plans])

  const executeCompare = async ({ showError = true } = {}) => {
    if (selectedPlanIds.length < 2 || selectedDimensions.length === 0) {
      setCompareData(null)
      if (showError) {
        setLoadError('请至少选择 2 个计划和 1 个维度后再比较')
      }
      return
    }

    try {
      setCompareBusy(true)
      if (showError) setLoadError('')
      const data = await runCompare({
        plan_ids: selectedPlanIds,
        dimensions: selectedDimensions,
        filters: {},
      })
      setCompareData(data)
    } catch (error) {
      if (showError) {
        setLoadError(error?.response?.data?.detail || '比较请求失败')
      }
    } finally {
      setCompareBusy(false)
    }
  }

  useEffect(() => {
    let mounted = true
    async function bootstrap() {
      try {
        setBusy(true)
        let [planList, dims] = await Promise.all([fetchPlans(), fetchDimensions()])
        if (planList.length === 0) {
          await runIngestion()
          planList = await fetchPlans()
        }
        if (!mounted) return
        setPlans(planList)
        setDimensionDefs(dims)

        const session = await createSession()
        if (!mounted) return
        setSessionId(session.session_id)
        setSelectedPlanIds(session.selected_plans || [])
        setSelectedDimensions(session.dimensions || [])
      } catch (error) {
        setLoadError(error?.response?.data?.detail || '加载失败，请检查后端服务')
      } finally {
        if (mounted) setBusy(false)
      }
    }
    bootstrap()
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    // 保留自动刷新，满足“选择后自动生成比较表”的需求。
    executeCompare({ showError: false })
  }, [selectedPlanIds, selectedDimensions])

  const onAskChat = async (content) => {
    if (!sessionId) return
    try {
      const result = await postChatMessage({ session_id: sessionId, content })
      setChatTurns(result.turns || [])
      setSelectedPlanIds(result.state.selected_plans || [])
      setSelectedDimensions(result.state.dimensions || [])
      if (result.compare) setCompareData(result.compare)
    } catch (error) {
      setLoadError(error?.response?.data?.detail || '聊天请求失败')
    }
  }

  const onIngest = async () => {
    try {
      setBusy(true)
      const result = await runIngestion()
      setToast(`已重建数据: 计划 ${result.plans_processed}，chunks ${result.chunks_written}，facts ${result.facts_written}`)
      const refreshedPlans = await fetchPlans()
      setPlans(refreshedPlans)
    } catch (error) {
      setLoadError(error?.response?.data?.detail || '重建数据失败')
    } finally {
      setBusy(false)
      window.setTimeout(() => setToast(''), 4000)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 via-teal-50 to-amber-50 text-ink">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">危疾保险产品并列比较</h1>
            <p className="mt-1 text-sm text-slate-600">动态筛选维度、差异高亮、聊天连续追问</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onIngest}
              className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-white hover:bg-teal-700"
            >
              重新解析数据
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1600px] grid-cols-1 gap-4 px-4 py-4 lg:grid-cols-[320px_minmax(0,1fr)_360px]">
        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <PlanDimensionPanel
            plans={plans}
            dimensionDefs={dimensionDefs}
            selectedPlanIds={selectedPlanIds}
            selectedDimensions={selectedDimensions}
            setSelectedPlanIds={setSelectedPlanIds}
            setSelectedDimensions={setSelectedDimensions}
            onRunCompare={() => executeCompare({ showError: true })}
            compareBusy={compareBusy}
          />
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <CompareTable
            planIds={selectedPlanIds}
            planMap={planMap}
            compareData={compareData}
            isLoading={busy || compareBusy}
          />
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <ChatPanel chatTurns={chatTurns} onAsk={onAskChat} />
        </section>
      </main>

      {(loadError || toast) && (
        <div className="fixed bottom-4 right-4 max-w-[420px] space-y-2">
          {loadError && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{loadError}</div>}
          {toast && <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">{toast}</div>}
        </div>
      )}
    </div>
  )
}

export default App
