<template>
  <q-page class="q-pa-md">
    <AwaitServer v-if="loading || error" :loading="loading" :error="error" @retry="retry" />
    <div v-else class="cloud-wrap">
      <q-img
        :src="img"
        loading="eager"
        fit="contain"
        no-transition
        spinner-color="primary"
        class="cloud-img"
      />
    </div>
  </q-page>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { backendURL } from '../data/lookups.js'
import { useSavedTabs } from '../stores/savedTabs'
import AwaitServer from '../components/AwaitServer.vue'
import { blobToDataURL } from '../utils/response'

const route = useRoute()
const store = useSavedTabs()

const loading = ref(true)
const error = ref('')
const img = ref('')

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
  img.value = ''
  try {
    const ids = parseIds()
    if (!ids.length) throw new Error('Пустой список идентификаторов')

    const res = await fetch(`${backendURL}/cloud`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'image/png,application/json',
      },
      body: JSON.stringify({ unique_keys: ids }),
    })

    if (!res.ok) {
      let msg = `HTTP ${res.status}`
      try {
        const j = await res.json()
        if (j?.error || j?.detail) msg += `: ${j.error || JSON.stringify(j.detail)}`
      } catch {
        // ignore
      }
      throw new Error(msg)
    }

    const ct = res.headers.get('content-type') || ''
    let url = ''

    if (ct.includes('image/png')) {
      const blob = await res.blob()
      url = await blobToDataURL(blob)
    } else if (ct.includes('application/json')) {
      const data = await res.json()
      url = data?.image || data?.images?.[0] || ''
    } else {
      throw new Error('Ожидали PNG или JSON от сервера')
    }

    if (!url) throw new Error('Пустое изображение облака')

    img.value = url
    store.addImages({
      title: `Облако тегов (${new Date().toLocaleString()})`,
      images: [url],
    })
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
.cloud-wrap {
  max-width: 1400px;
  margin: 0 auto;
}
.cloud-img {
  width: 100%;
  height: auto;
  display: block;
  object-fit: contain;
}
</style>
