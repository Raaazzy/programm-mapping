<template>
  <q-card flat bordered class="course-card">
    <q-card-section class="card-section">
      <div class="card-main">
        <div class="card-header">
          <q-checkbox
            :model-value="checked"
            @update:model-value="$emit('update:checked', $event)"
          />
          <div
            class="title ellipsis cursor-pointer"
            :title="c.title"
            @click="show = true"
          >
            {{ c.title }}
          </div>
        </div>

        <div class="tags">
          <q-chip v-for="tg in c.tags.slice(0, 3)" :key="tg" dense>{{ tg }}</q-chip>
        </div>
      </div>
    </q-card-section>

    <q-dialog v-model="show">
      <q-card style="max-width: 760px; width: 100%">
        <q-card-section class="row items-center justify-between">
          <div class="text-h6">{{ c.title }}</div>
          <q-btn icon="close" flat round dense v-close-popup />
        </q-card-section>
        <q-separator />

        <q-card-section class="q-gutter-sm">
          <div><strong>Кто читает:</strong> {{ c.department || '—' }}</div>
          <div><strong>Когда читается:</strong> {{ c.when || '—' }}</div>
          <div><strong>Охват аудитории:</strong> {{ c.audience || '—' }}</div>
          <div><strong>Язык:</strong> {{ c.language || '—' }}</div>

          <div class="text-subtitle2 q-mt-md">Аннотация:</div>
          <div class="prewrap">{{ c.annotation || '—' }}</div>

          <div v-if="c.outcomes.length" class="text-subtitle2 q-mt-md">Планируемые результаты обучения:</div>
          <ul v-if="c.outcomes.length" class="q-pl-md">
            <li v-for="(item, i) in c.outcomes" :key="i">{{ item }}</li>
          </ul>

          <div v-if="c.controls.length" class="text-subtitle2 q-mt-md">Элементы контроля:</div>
          <ul v-if="c.controls.length" class="q-pl-md">
            <li v-for="(item, i) in c.controls" :key="i">{{ item }}</li>
          </ul>
        </q-card-section>
      </q-card>
    </q-dialog>
  </q-card>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  course: { type: Object, required: true },
  checked: { type: Boolean, default: false }
})

const show = ref(false)

function parseArrayish (val) {
  if (Array.isArray(val)) return val
  if (typeof val === 'string') {
    const s = val.trim()
    if (s.startsWith('[') && s.endsWith(']')) {
      try { return JSON.parse(s.replace(/'/g, '"')) } catch {console.log("error")}
      return s.slice(1, -1)
        .split(',')
        .map(x => x.trim().replace(/^['"]|['"]$/g, ''))
        .filter(Boolean)
    }
    return s ? [s] : []
  }
  return val != null ? [String(val)] : []
}


const c = computed(() => {
  const raw = props.course || {}
  const id = raw['Уникальный ключ'] ?? raw.id ?? raw.key
  const title = raw['Название дисциплины'] ?? raw.title ?? raw.name ?? raw.label ?? 'Без названия'
  const faculty = raw['Факультет'] ?? raw.faculty ?? null
  const program = raw['ОП'] ?? raw.program ?? null
  const course = raw['Курс'] ?? raw.course ?? null
  const modules = parseArrayish(raw['Модули'] ?? raw.modules ?? [])

  const tags = parseArrayish(raw['Ключевые слова'] ?? raw.tags ?? raw.keywords ?? [])
    .map(String)

  const languages = parseArrayish(raw['Языки'] ?? raw.language ?? raw.lang ?? [])
  const department = raw['Департамент'] ?? raw.department ?? raw.whoReads ?? raw.lecturer ?? raw.readBy ?? raw.departmentName
  const audience = raw['Охват'] ?? raw.audience ?? raw.coverage ?? raw.scope
  const annotation = raw['Аннотация'] ?? raw.annotation ?? raw.summary ?? raw.description ?? raw.desc
  const outcomes = parseArrayish(raw['Результаты'] ?? raw.outcomes ?? raw.learningOutcomes ?? raw.results ?? [])
  const controls = parseArrayish(raw['Контроли'] ?? raw.controls ?? raw.assessment ?? raw.controlElements ?? [])

  const when = raw['Когда читается'] ?? (() => {
    const parts = []
    if (course != null && course !== '') parts.push(`${course}-й курс`)
    if (modules.length) parts.push(`${modules.join(', ')} модул${modules.length > 1 ? 'я' : 'ь'}`)
    return parts.join(', ')
  })()

  return {
    id, title, tags, faculty, program, course,
    modules, languages, department, audience,
    annotation, outcomes, controls, when,
    language: languages.join(', ')
  }
})
</script>


<style scoped>
.course-card {
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  overflow: hidden;
  min-height: 104px;        
}


.card-section {
  display: flex;
  padding: 12px 16px;
  overflow: hidden;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}


.title {
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}


.tags {
  display: flex;
  flex-wrap: nowrap;    
  gap: 6px;
  padding-top: 4px;
  overflow: hidden;     
  max-height: 28px;     
}

</style>
