<template>
  <q-page class="q-pa-md">
    <AwaitServer v-if="loading || error" :loading="loading" :error="error" @retry="retry" />
    <div v-else class="dash-columns">
      <!-- колонка 1: 1,2,3,4 -->
      <div class="dash-col">
        <div v-for="img in col1" :key="img.key" class="dash-card">
          <q-img
            :src="img.src"
            fit="contain"
            loading="eager"
            no-transition
            spinner-color="primary"
            class="dash-img"
          />
        </div>
      </div>

      <!-- колонка 2: только 5 (большая) -->
      <div class="dash-col">
        <div v-if="col2" class="dash-card dash-card--large">
          <q-img
            :src="col2.src"
            fit="contain"
            loading="eager"
            no-transition
            spinner-color="primary"
            class="dash-img dash-img--large"
          />
        </div>
      </div>

      <!-- колонка 3: только 6 (большая) -->
      <div class="dash-col">
        <div v-if="col3" class="dash-card dash-card--large">
          <q-img
            :src="col3.src"
            fit="contain"
            loading="eager"
            no-transition
            spinner-color="primary"
            class="dash-img dash-img--large"
          />
        </div>
      </div>
    </div>
  </q-page>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { backendURL } from '../data/lookups.js'
import { useSavedTabs } from '../stores/savedTabs'
import AwaitServer from '../components/AwaitServer.vue'

const route = useRoute()
const store = useSavedTabs()

const loading = ref(true)
const error = ref('')
const images = ref([])

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
  images.value = []
  try {
    const ids = parseIds()
    if (!ids.length) throw new Error('Пустой список идентификаторов')

    const res = await fetch(`${backendURL}/dashboards`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
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
    const data = await res.json()
    const charts = data?.charts || {}

    const arr = Object.entries(charts).map(([key, b64]) => {
      const clean = String(b64 || '').replace(/\s+/g, '')
      return { key, src: clean.startsWith('data:') ? clean : `data:image/png;base64,${clean}` }
    })
    images.value = arr

    store.addImages({
      title: `Дашборды (${new Date().toLocaleString()})`,
      images: arr.map((x) => x.src),
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

/* ----- раскладка по колонкам ----- */
const byKey = computed(() => Object.fromEntries(images.value.map((i) => [i.key, i])))

// 1: control_distribution_type, 2: control_distribution_block, 3: control_distribution_format, 4: control_distribution_exam
const col1Order = [
  'control_distribution_type',
  'control_distribution_block',
  'control_distribution_format',
  'control_distribution_exam',
]
const col1 = computed(() => col1Order.map((k) => byKey.value[k]).filter(Boolean))

// 5: control_distribution_number (большой)
const col2 = computed(() => byKey.value['control_distribution_number'] || null)

// 6: control_distribution_name (большой)
const col3 = computed(() => byKey.value['control_distribution_name'] || null)
</script>

<style scoped>
.dash-columns {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
  align-items: start;
}

.dash-col {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.dash-card {
  background: #fff;
  border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
  padding: 12px;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

/* изображение */
.dash-img {
  width: 100%;
  height: auto;
  object-fit: contain;
  display: block;
}

/* для 5 и 6 — увеличиваем */
.dash-card--large {
  padding: 12px;
}

.dash-card:nth-child(5) .dash-img,
.dash-card:nth-child(6) .dash-img {
  width: 100%;
  height: auto;
  transform: scale(4);
  transform-origin: top left;
  margin-bottom: 20px;
}

/* адаптив: на узких экранах стекаем 1 колонкой, затем 2 */
@media (max-width: 1400px) {
  .dash-img--large {
    max-height: 60vh;
  }
}
@media (max-width: 1100px) {
  .dash-columns {
    flex-wrap: wrap;
  }
  .dash-col {
    flex: 1 1 100%;
  }
}
@media (min-width: 1101px) and (max-width: 1400px) {
  .dash-columns {
    flex-wrap: wrap;
  }
  .dash-col {
    flex: 1 1 calc(50% - 8px);
  } /* 2 колонки */
}
</style>
