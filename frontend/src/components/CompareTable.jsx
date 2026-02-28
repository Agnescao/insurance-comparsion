import React from 'react'

function CompareTable({ planIds, planMap, compareData, isLoading }) {
  if (isLoading) {
    return <div className="text-sm text-slate-500">处理中...</div>
  }

  if (planIds.length < 2) {
    return <div className="text-sm text-slate-500">请先在左侧选择至少两个计划。</div>
  }

  if (!compareData?.rows?.length) {
    return <div className="text-sm text-slate-500">暂无可展示对比数据。</div>
  }

  return (
    <div className="overflow-auto">
      <table className="w-full min-w-[900px] border-collapse">
        <thead>
          <tr className="bg-slate-100">
            <th className="sticky left-0 z-10 border border-slate-200 bg-slate-100 px-3 py-2 text-left text-sm">比较维度</th>
            {planIds.map((planId) => (
              <th key={planId} className="border border-slate-200 px-3 py-2 text-left text-sm">
                {planMap[planId]?.name || planId}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {compareData.rows.map((row) => {
            const distinct = new Set(
              planIds
                .map((pid) => row.plan_values?.[pid]?.value || '')
                .filter((v) => v && v !== '未提取到')
                .map((v) => v.trim().toLowerCase()),
            )
            return (
              <tr key={row.dimension_key} className={row.is_different ? 'bg-amber-50/60' : ''}>
                <td className="sticky left-0 z-10 border border-slate-200 bg-white px-3 py-2 align-top text-sm font-semibold">
                  <div>{row.dimension_label}</div>
                  {row.is_different && <div className="mt-1 text-xs font-medium text-amber-700">差异明显</div>}
                </td>
                {planIds.map((planId) => {
                  const cell = row.plan_values?.[planId]
                  const val = cell?.value || '未提取到'
                  const isCellDiff = row.is_different && distinct.size > 1 && val !== '未提取到'
                  return (
                    <td
                      key={`${row.dimension_key}-${planId}`}
                      className={`border border-slate-200 px-3 py-2 align-top text-sm ${isCellDiff ? 'font-semibold text-slate-900' : 'text-slate-700'}`}
                    >
                      <div>{val}</div>
                      {cell?.source && (
                        <div className="mt-2 text-xs text-slate-500">
                          来源: {cell.source.section || 'N/A'}
                          {cell.source.page ? ` - 第 ${cell.source.page} 页` : ''}
                        </div>
                      )}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default CompareTable
