import React, { useState } from 'react'

function ChatPanel({ chatTurns, onAsk }) {
  const [content, setContent] = useState('')
  const [sending, setSending] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    const text = content.trim()
    if (!text || sending) return
    setSending(true)
    try {
      await onAsk(text)
      setContent('')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex h-[78vh] flex-col">
      <h2 className="text-lg font-semibold">聊天面板</h2>
      <p className="mb-3 text-xs text-slate-500">连续提问会自动保持已选计划和维度</p>

      <div className="flex-1 space-y-3 overflow-auto rounded-md border border-slate-200 bg-slate-50 p-3">
        {chatTurns.length === 0 && <div className="text-sm text-slate-500">示例: “加入卵巢癌场景比较并告诉我差异”</div>}
        {chatTurns.map((turn, idx) => (
          <div key={`${turn.timestamp}-${idx}`} className={turn.role === 'assistant' ? 'rounded-md bg-white p-2' : 'rounded-md bg-teal-50 p-2'}>
            <div className="text-xs font-semibold uppercase text-slate-500">{turn.role}</div>
            <div className="whitespace-pre-wrap text-sm leading-6">{turn.content}</div>
          </div>
        ))}
      </div>

      <form onSubmit={submit} className="mt-3 space-y-2">
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="输入问题，例如：请加入除外责任和自付额维度"
          className="h-24 w-full resize-none rounded-md border border-slate-300 px-3 py-2 text-sm outline-none ring-accent focus:ring"
        />
        <button
          type="submit"
          disabled={sending}
          className="w-full rounded-md bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-60"
        >
          {sending ? '发送中...' : '发送'}
        </button>
      </form>
    </div>
  )
}

export default ChatPanel
