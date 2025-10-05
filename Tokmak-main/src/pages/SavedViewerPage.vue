<template>
  <q-page class="q-pa-none page-bleed">
    <div v-if="!tab" class="q-pa-md text-grey">Сохранённая вкладка не найдена</div>

    <!-- HTML -->
    <div v-else-if="tab.type === 'html'" class="frame-wrap">
      <iframe class="html-frame" :srcdoc="srcdoc"></iframe>
    </div>

    <!-- Изображения -->
    <div v-else-if="tab.type === 'images'" class="q-pa-md">
      <!-- если одно изображение - для облака тегов -->
      <div v-if="isSingleImage" class="cloud-wrap">
        <q-img
          :src="onlyImage"
          fit="contain"
          loading="eager"
          no-transition
          spinner-color="primary"
          class="cloud-img"
        />
      </div>

      <!-- если несколько изображений - для дашборда -->
      <div v-else class="dash-columns">
        <!-- колонка 1 -->
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

        <!-- колонка 2 -->
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

        <!-- колонка 3 -->
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
    </div>
  </q-page>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useSavedTabs } from '../stores/savedTabs'
import { backendURL } from '../data/lookups.js'
import { prepareVisIframeHtml } from '../utils/html'

const route = useRoute()
const store = useSavedTabs()

const tab = computed(() => store.byId(route.params.id))

const srcdoc = computed(() => {
  const t = tab.value
  if (!t || t.type !== 'html') return ''
  return prepareVisIframeHtml(t.payload?.html || '', backendURL)
})

const imgs = computed(() => tab.value?.payload?.images || [])

const isSingleImage = computed(() => imgs.value.length === 1)
const onlyImage = computed(() => imgs.value[0] || '')

const col2 = computed(() => (imgs.value[1] ? { key: 'c2_1', src: imgs.value[1] } : null))
const col3 = computed(() => (imgs.value[5] ? { key: 'c3_5', src: imgs.value[5] } : null))
const col1 = computed(() =>
  imgs.value
    .map((src, idx) => ({ idx, src }))
    .filter(({ idx }) => idx !== 1 && idx !== 5)
    .map(({ idx, src }) => ({ key: `c1_${idx}`, src })),
)
</script>

<style scoped>
.page-bleed {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: auto;
  padding-bottom: 64px;
}
.frame-wrap {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
}
.html-frame {
  flex: 1 1 auto;
  width: 100%;
  height: 100%;
  display: block;
  border: 0;
  background: white;
}
.images-wrap {
  width: 100%;
  max-width: 1600px;
  margin: 0 auto;
}

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
.dash-img {
  width: 100%;
  height: auto;
  object-fit: contain;
  display: block;
}

.dash-card--large {
  padding: 12px;
}
.dash-img--large {
  width: 100%;
  height: auto;
  object-fit: contain;
  display: block;
}

.cloud-wrap {
  max-width: 1600px;
  margin: 0 auto;
}
.cloud-img {
  width: 100%;
  height: 75vh;
  object-fit: contain;
  display: block;
}

@media (max-width: 1400px) {
  .dash-columns {
    grid-template-columns: repeat(2, 1fr);
  }
  .dash-img--large {
    height: 60vh;
  }
}
@media (max-width: 900px) {
  .dash-columns {
    grid-template-columns: 1fr;
  }
  .dash-img--large {
    height: 50vh;
  }
}
</style>
