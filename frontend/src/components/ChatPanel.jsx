import React, { useState } from 'react'

function renderHighlightedText(text) {
  const lines = String(text || '').split('\n')
  return lines.map((line, lineIndex) => {
    const segments = line.split(/(\*\*[^*]+\*\*)/g)
    return (
      <div key={`line-${lineIndex}`}>
        {segments.map((seg, segIndex) => {
          const isMarked = seg.startsWith('**') && seg.endsWith('**') && seg.length > 4
          if (!isMarked) {
            return <span key={`seg-${segIndex}`}>{seg}</span>
          }
          const value = seg.slice(2, -2)
          return (
            <mark key={`seg-${segIndex}`} className="rounded bg-amber-200 px-1 font-semibold text-slate-900">
              {value}
            </mark>
          )
        })}
      </div>
    )
  })
}

function ChatPanel({ chatTurns, streamingReply, onAsk }) {
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
      <p className="mb-3 text-xs text-slate-500">连续提问会自动保持已选计划和维度，并流式输出结论</p>

      <div className="flex-1 space-y-3 overflow-auto rounded-md border border-slate-200 bg-slate-50 p-3">
        {chatTurns.length === 0 && !streamingReply && (
          <div className="text-sm text-slate-500">示例: “哪个计划在我患卵巢癌时提供更好的保障？”</div>
        )}

        {chatTurns.map((turn, idx) => (
          <div key={`${turn.timestamp}-${idx}`} className={turn.role === 'assistant' ? 'rounded-md bg-white p-2' : 'rounded-md bg-teal-50 p-2'}>
            <div className="text-xs font-semibold uppercase text-slate-500">{turn.role}</div>
            <div className="text-sm leading-6">{renderHighlightedText(turn.content)}</div>
          </div>
        ))}

        {streamingReply && (
          <div className="rounded-md bg-white p-2">
            <div className="text-xs font-semibold uppercase text-slate-500">assistant</div>
            <div className="text-sm leading-6">{renderHighlightedText(streamingReply)}</div>
          </div>
        )}
      </div>

      <form onSubmit={submit} className="mt-3 space-y-2">
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="输入问题，例如：哪个计划在我患卵巢癌时提供更好的保障？"
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
