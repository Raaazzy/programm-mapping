<template>
  <aside class="q-pa-md q-gutter-md">
    <div class="text-subtitle2 q-mb-sm">Фильтры</div>

    <q-select
      v-model="local.faculty"
      :options="filteredFaculty"
      label="Факультет"
      dense
      outlined
      clearable
      use-input
      input-debounce="0"
      input-class="ellipsis text-no-wrap"
      @filter="onFilterFaculty"
      @update:model-value="emitChange"
    >
      <template #no-option>
        <q-item><q-item-section class="text-grey">Нет результатов</q-item-section></q-item>
      </template>
      <template #option="scope">
        <q-item v-bind="scope.itemProps">
          <q-item-section><div class="ellipsis">{{ scope.opt }}</div></q-item-section>
        </q-item>
      </template>
    </q-select>

    <q-select
      v-model="local.program"
      :options="filteredProgram"
      label="Программа"
      dense
      outlined
      clearable
      use-input
      input-debounce="0"
      input-class="ellipsis text-no-wrap"
      @filter="onFilterProgram"
      @update:model-value="emitChange"
    >
      <template #no-option>
        <q-item><q-item-section class="text-grey">Нет результатов</q-item-section></q-item>
      </template>
      <template #option="scope">
        <q-item v-bind="scope.itemProps">
          <q-item-section><div class="ellipsis">{{ scope.opt }}</div></q-item-section>
        </q-item>
      </template>
    </q-select>

    <q-select
      v-model="local.course"
      :options="filteredCourse"
      label="Курс"
      dense
      outlined
      clearable
      use-input
      input-debounce="0"
      input-class="ellipsis text-no-wrap"
      @filter="onFilterCourse"
      @update:model-value="emitChange"
    >
      <template #no-option>
        <q-item><q-item-section class="text-grey">Нет результатов</q-item-section></q-item>
      </template>
      <template #option="scope">
        <q-item v-bind="scope.itemProps">
          <q-item-section><div class="ellipsis">{{ scope.opt }}</div></q-item-section>
        </q-item>
      </template>
    </q-select>

    <q-select
      v-model="local.module"
      :options="filteredModule"
      label="Модуль"
      dense
      outlined
      clearable
      use-input
      input-debounce="0"
      input-class="ellipsis text-no-wrap"
      @filter="onFilterModule"
      @update:model-value="emitChange"
    >
      <template #no-option>
        <q-item><q-item-section class="text-grey">Нет результатов</q-item-section></q-item>
      </template>
      <template #option="scope">
        <q-item v-bind="scope.itemProps">
          <q-item-section><div class="ellipsis">{{ scope.opt }}</div></q-item-section>
        </q-item>
      </template>
    </q-select>
  </aside>
</template>

<script setup>
import { reactive, watch, ref } from 'vue'
import { faculties, programs, courses, modules } from '../data/lookups.js'

const props = defineProps({
  modelValue: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['update:modelValue', 'apply'])

const defaults = { faculty: null, program: null, course: null, module: null }
const local = reactive({ ...defaults, ...props.modelValue })

watch(
  () => props.modelValue,
  (val) => {
    if (!val) return
    Object.assign(local, { ...defaults, ...val })
  },
  { deep: true }
)

const baseFaculty = ref([...(faculties || [])])
const baseProgram = ref([...(programs || [])])
const baseCourse = ref([...(courses || [])])
const baseModule = ref([...(modules || [])])

const filteredFaculty = ref([...baseFaculty.value])
const filteredProgram = ref([...baseProgram.value])
const filteredCourse = ref([...baseCourse.value])
const filteredModule = ref([...baseModule.value])

watch(baseFaculty, (v) => (filteredFaculty.value = [...v]))
watch(baseProgram, (v) => (filteredProgram.value = [...v]))
watch(baseCourse, (v) => (filteredCourse.value = [...v]))
watch(baseModule, (v) => (filteredModule.value = [...v]))

const startsWithCi = (a, b) => String(a).toLowerCase().startsWith(String(b).toLowerCase())

function makeFilter (base, target) {
  return (val, update) => {
    update(() => {
      if (!val) target.value = [...base.value]
      else target.value = base.value.filter((opt) => startsWithCi(opt, val))
    })
  }
}

const onFilterFaculty = makeFilter(baseFaculty, filteredFaculty)
const onFilterProgram = makeFilter(baseProgram, filteredProgram)
const onFilterCourse = makeFilter(baseCourse, filteredCourse)
const onFilterModule = makeFilter(baseModule, filteredModule)

function emitChange () {
  const payload = { ...local }
  emit('update:modelValue', payload)
  emit('apply', payload)
}
</script>

<style scoped>
aside {
  width: 300px;
  flex: 0 0 300px;
}

:deep(.q-field__native) {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

:deep(.q-item__label),
:deep(.q-item .q-item__section > div) {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
