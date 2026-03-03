import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
})

export async function fetchPlans() {
  const { data } = await api.get('/api/plans')
  return data
}

export async function fetchDimensions() {
  const { data } = await api.get('/api/dimensions')
  return data
}

export async function runCompare(payload) {
  const { data } = await api.post('/api/compare', payload, { timeout: 10000 })
  return data
}

export async function createSession() {
  const { data } = await api.post('/api/chat/session', {})
  return data
}

export async function postChatMessage(payload) {
  const { data } = await api.post('/api/chat/message', payload)
  return data
}

export async function postChatMessageStream(payload, handlers = {}) {
  const { onToken, onDone, onError } = handlers
  const url = `${API_BASE}/api/chat/message/stream`
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!response.ok || !response.body) {
    const text = await response.text()
    throw new Error(text || `stream request failed: ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const emitEvent = (eventBlock) => {
    let eventName = 'message'
    const dataLines = []

    // Normal SSE block with real newlines.
    if (eventBlock.includes('\n')) {
      const lines = eventBlock.split('\n')
      lines.forEach((line) => {
        const normalized = line.replace(/\r$/, '')
        if (normalized.startsWith('event:')) {
          eventName = normalized.slice(6).trim()
        } else if (normalized.startsWith('data:')) {
          dataLines.push(normalized.slice(5).trim())
        }
      })
    } else if (eventBlock.includes('\\ndata:')) {
      // Escaped SSE block fallback: "event: token\\ndata: {...}".
      const eventIdx = eventBlock.indexOf('event:')
      const dataIdx = eventBlock.indexOf('\\ndata:')
      if (eventIdx >= 0 && dataIdx > eventIdx) {
        eventName = eventBlock.slice(eventIdx + 6, dataIdx).trim()
        const payloadText = eventBlock.slice(dataIdx + '\\ndata:'.length).trim()
        if (payloadText) dataLines.push(payloadText)
      }
    }

    if (!dataLines.length) return
    let payloadObj
    try {
      payloadObj = JSON.parse(dataLines.join('\n'))
    } catch {
      payloadObj = { text: dataLines.join('\n') }
    }

    if (eventName === 'token' && onToken) onToken(payloadObj)
    if (eventName === 'done' && onDone) onDone(payloadObj)
  }

  const drainBuffer = () => {
    // Prefer standard SSE framing.
    const normalized = buffer.replace(/\r\n/g, '\n')
    if (normalized.includes('\n\n')) {
      const parts = normalized.split('\n\n')
      buffer = parts.pop() || ''
      parts.forEach(emitEvent)
      return
    }

    // Fallback for escaped framing.
    if (buffer.includes('\\n\\n')) {
      const parts = buffer.split('\\n\\n')
      buffer = parts.pop() || ''
      parts.forEach(emitEvent)
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      drainBuffer()
    }

    if (buffer.trim()) {
      emitEvent(buffer)
    }
  } catch (error) {
    if (onError) onError(error)
    throw error
  }
}

export async function runIngestion() {
  const { data } = await api.post('/api/ingest/run')
  return data
}
