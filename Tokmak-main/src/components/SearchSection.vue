<template>
  <div class="search-with-tags">
    <q-input
      v-model="localSearch"
      outlined
      dense
      placeholder="Поиск"
      clearable
      @update:model-value="onSearchChange"
      @keyup.enter="emitSubmit"
    >
      <template #prepend>
        <q-icon name="search" />
      </template>
    </q-input>

    <div class="q-mt-md row items-start q-col-gutter-xs">
      <q-btn
        v-for="t in TAGS"
        :key="t"
        :label="t"
        no-caps
        dense
        size="md"
        padding="xs md"
        rounded
        :unelevated="!isSelected(t)"
        :outline="!isSelected(t)"
        :color="isSelected(t) ? 'primary' : 'grey-2'"
        :text-color="isSelected(t) ? 'white' : 'grey-8'"
        class="q-ma-xs text-caption"
        @click="toggleTag(t)"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

import { tags as TAGS } from '../data/lookups.js'

const props = defineProps({
  modelValue: { type: String, default: '' },
  tags: { type: Array, default: () => [] },
})
const emit = defineEmits(['update:modelValue', 'update:tags', 'submit', 'change'])

const localSearch = ref(props.modelValue)
const selected = ref(new Set(props.tags))

watch(
  () => props.modelValue,
  (v) => {
    if (v !== localSearch.value) localSearch.value = v
  },
)
watch(
  () => props.tags,
  (v) => {
    selected.value = new Set(v)
  },
)

function onSearchChange(v) {
  emit('update:modelValue', v)
  emitChange()
}

function toggleTag(val) {
  const s = selected.value
  s.has(val) ? s.delete(val) : s.add(val)
  emit('update:tags', Array.from(s))
  emitChange()
}

function isSelected(val) {
  return selected.value.has(val)
}

function emitSubmit() {
  emit('submit', { search: localSearch.value, tags: Array.from(selected.value) })
}

function emitChange() {
  emit('change', { search: localSearch.value, tags: Array.from(selected.value) })
}
</script>

<style scoped>
.search-with-tags {
  width: 100%;
  max-width: 920px;
}
</style>
