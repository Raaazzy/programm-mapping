<template>
  <section class="q-pa-md courses-list">
    <div class="row items-center justify-between q-mb-sm sticky-actions">
      <div class="text-subtitle2">Курсы</div>
      <div class="row q-gutter-sm">
        <q-btn
          size="sm"
          flat
          icon="done_all"
          label="выделить все"
          @click="selectAll"
          :disable="filtered.length === 0"
        />
        <q-btn
          size="sm"
          flat
          color="negative"
          icon="restart_alt"
          label="сбросить"
          @click="resetAll"
          :disable="isEmptySelection"
        />
      </div>
    </div>

    <q-separator class="q-mb-sm" />

    <div v-if="filtered.length === 0" class="text-grey q-pa-md">Ничего не найдено</div>

    <div v-else class="list-body">
      <!-- КРИТИЧНО: у QVirtualScroll должна быть явная высота -->
      <q-virtual-scroll
        class="course-vs"
        style="height: 100%"
        :items="filtered"
        :virtual-scroll-item-size="CARD_SIZE"
        :virtual-scroll-slice-size="64"
        :virtual-scroll-slice-ratio="3"
        v-slot="{ item }"
      >
        <CourseCard
          :course="item"
          :checked="!!selectionMap[item.id]"
          @update:checked="onToggle(item.id, $event)"
          class="q-mb-sm"
        />
      </q-virtual-scroll>
    </div>
  </section>
</template>

<script setup>
import { shallowRef, computed, reactive, watch, onMounted } from 'vue'
import { coursesList } from '../data/lookups.js'
import CourseCard from './CourseCard.vue'
const CARD_SIZE = 104

const props = defineProps({
  modelValue: { type: Array, default: () => [] },
  filters: {
    type: Object,
    default: () => ({ faculty: null, program: null, course: null, module: null }),
  },
  search: { type: String, default: '' },
  activeTags: { type: Array, default: () => [] },
  height: { type: String, default: '100%' },
})

const emit = defineEmits(['update:modelValue', 'reset'])

function parseList(val) {
  if (val == null) return []
  if (Array.isArray(val)) return val
  if (typeof val === 'string') {
    const s = val.trim()
    if (s[0] === '[' && s[s.length - 1] === ']') {
      const inner = s.slice(1, -1).trim()
      if (!inner) return []
      return inner.split(',').map((x) => x.trim().replace(/^['"]|['"]$/g, ''))
    }
    return [val]
  }
  return [String(val)]
}

const catalog = shallowRef([])

onMounted(() => {
  const src = Array.isArray(coursesList) ? coursesList : []
  catalog.value = src.map((raw, i) => {
    const id = String(raw['Уникальный ключ'] ?? i)
    const title = String(raw['Название дисциплины'] ?? '')
    const tags = parseList(raw['Ключевые слова']).map(String)

    const modules = parseList(raw['Модули']).map(String)
    const modulesSet = new Set(modules)

    const faculty = raw['Факультет'] ?? null
    const program = raw['ОП'] ?? null
    const course = raw['Курс'] ?? null

    return {
      id,
      title,
      tags,
      faculty,
      program,
      course,
      modules,
      modulesSet,

      lc_title: title.toLowerCase(),
      lc_tags: tags.map((t) => t.toLowerCase()),

      department: raw['Департамент'] ?? null,
      audience: raw['Охват'] ?? null,
      languages: parseList(raw['Языки']),
      annotation: raw['Аннотация'] ?? null,
      outcomes: parseList(raw['Результаты']),
      controls: parseList(raw['Контроли']),
    }
  })
})


const filtered = computed(() => {
  const f = props.filters || {}
  const hasFaculty = !!f.faculty
  const hasProgram = !!f.program
  const hasCourse = f.course != null && f.course !== ''
  const hasModule = !!f.module

  const q = (props.search || '').trim().toLowerCase()
  const hasQ = q.length > 0

  const tagsArray = props.activeTags || []
  const hasTags = tagsArray.length > 0
  const tagsSet = hasTags ? new Set(tagsArray.map(String)) : null

  const moduleKey = hasModule ? String(f.module) : null
  const courseKey = hasCourse ? String(f.course) : null

  return catalog.value.filter((c) => {
    if (hasFaculty && c.faculty !== f.faculty) return false
    if (hasProgram && c.program !== f.program) return false
    if (hasCourse && String(c.course) !== courseKey) return false
    if (hasModule && !c.modulesSet.has(moduleKey)) return false

    if (hasTags) {
      let ok = false
      const len = c.tags.length
      for (let i = 0; i < len; i++) {
        if (tagsSet.has(String(c.tags[i]))) {
          ok = true
          break
        }
      }
      if (!ok) return false
    }

    if (hasQ) {
      if (c.lc_title.indexOf(q) === -1) {
        let hit = false
        const len2 = c.lc_tags.length
        for (let j = 0; j < len2; j++) {
          if (c.lc_tags[j].indexOf(q) !== -1) {
            hit = true
            break
          }
        }
        if (!hit) return false
      }
    }

    return true
  })
})

const selectionMap = reactive({})

watch(
  () => props.modelValue,
  (val) => {
    const m = {}
    for (let i = 0; i < val.length; i++) m[String(val[i])] = true
    for (const k of Object.keys(selectionMap)) delete selectionMap[k]
    Object.assign(selectionMap, m)
  },
  { immediate: true },
)

const selectedIds = computed(() => {
  const out = []
  for (const k in selectionMap) if (selectionMap[k]) out.push(k)
  return out
})

const isEmptySelection = computed(() => selectedIds.value.length === 0)

function onToggle(id, checked) {
  selectionMap[String(id)] = !!checked
  emit('update:modelValue', selectedIds.value)
}

function selectAll() {
  const list = filtered.value
  for (let i = 0; i < list.length; i++) selectionMap[list[i].id] = true
  emit('update:modelValue', selectedIds.value)
}

function resetAll() {
  for (const k of Object.keys(selectionMap)) selectionMap[k] = false
  emit('update:modelValue', [])
  emit('reset')
}

</script>

<style scoped>
.courses-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0; 
}

.sticky-actions {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--q-background, white);
  padding-top: 4px;
}

.list-body {
  flex: 1 1 auto; 
  min-height: 0; 
}


.course-vs {
  overflow-x: hidden;
}
</style>
