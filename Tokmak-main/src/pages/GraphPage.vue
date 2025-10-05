<template>
  <q-page class="q-pa-none graph-page window-height">
    <AwaitServer v-if="loading || error" :loading="loading" :error="error" @retry="retry" />
    <div v-else class="frame-wrap">
      <iframe class="html-frame" :srcdoc="html"></iframe>
    </div>
  </q-page>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { backendURL } from '../data/lookups.js'
import { useSavedTabs } from '../stores/savedTabs'
import AwaitServer from '../components/AwaitServer.vue'
import { prepareVisIframeHtml } from '../utils/html'

const route = useRoute()
const store = useSavedTabs()

const loading = ref(true)
const error = ref('')
const html = ref('')

function parseIds() {
  const raw = route.query.ids
  if (!raw) return []
  return String(raw)
    .split(',')
    .map((s) => Number(s.trim()))
    .filter(Number.isFinite)
}

async function postData() {
  loading.value = true
  error.value = ''
  html.value = ''
  try {
    const ids = parseIds()
    if (!ids.length) throw new Error('Пустой список идентификаторов')

    const res = await fetch(`${backendURL}/graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ unique_keys: ids }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const htmlString = await res.text() // <-- HTMLResponse => text()
    const wrapped = /<html[\s>]/i.test(htmlString) ? htmlString : prepareVisIframeHtml(htmlString)
    html.value = wrapped

    store.addHtml({ title: `Граф (${new Date().toLocaleString()})`, html: wrapped })
  } catch (e) {
    error.value = e?.message || String(e)
  } finally {
    loading.value = false
  }
}

function retry() {
  postData()
}

onMounted(postData)
</script>

<style scoped>
.graph-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}
.frame-wrap {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
}
.html-frame {
  flex: 1 1 auto;
  width: 100%;
  height: 100vh;
  display: block;
  border: 0;
  background: white;
}
</style>
