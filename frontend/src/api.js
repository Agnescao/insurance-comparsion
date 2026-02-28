import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
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
  const { data } = await api.post('/api/compare', payload)
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

export async function runIngestion() {
  const { data } = await api.post('/api/ingest/run')
  return data
}
