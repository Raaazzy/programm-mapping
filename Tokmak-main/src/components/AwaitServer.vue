<template>
  <div v-if="loading || error" class="center-wrap" :style="{ minHeight }">
    <template v-if="loading">
      <div class="text-h4 text-weight-medium">Ждём ответа сервера</div>
      <q-icon :name="icon" :size="iconSize" color="grey-6" class="q-mt-sm" />
      <div class="text-h5 text-mono">{{ elapsedText }}</div>
      <div class="text-body1 text-grey-7 hint">
        {{ hintText }}
      </div>
    </template>

    <template v-else>
      <div class="text-h5 text-negative q-mb-sm">Ошибка: {{ error }}</div>
      <q-btn color="primary" label="Повторить" @click="$emit('retry')" />
    </template>
  </div>
</template>

<script setup>
import { ref, computed, watch, onUnmounted } from 'vue'

const props = defineProps({
  loading: { type: Boolean, default: false },
  error:   { type: String,  default: ''   },
  icon:    { type: String,  default: 'hourglass_empty' },
  iconSize:{ type: String,  default: '96px' },
  minHeight: { type: String, default: 'calc(100vh - 160px)' },
  hintText: {
    type: String,
    default: 'Построение графиков по большим количеством дисциплин может занять много времени'
  }
})

const elapsedSec = ref(0)
let timerId = null

const elapsedText = computed(() => {
  const m = Math.floor(elapsedSec.value / 60).toString().padStart(2, '0')
  const s = (elapsedSec.value % 60).toString().padStart(2, '0')
  return `${m}:${s}`
})

watch(() => props.loading, (isLoading) => {
  if (isLoading) startTimer()
  else stopTimer()
}, { immediate: true })

function startTimer () { stopTimer(); elapsedSec.value = 0; timerId = setInterval(() => { elapsedSec.value++ }, 1000) }
function stopTimer  () { if (timerId) { clearInterval(timerId); timerId = null } }

onUnmounted(stopTimer)
</script>

<style scoped>
.center-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  text-align: center;
}
.text-mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
.hint { max-width: 680px; }
</style>
