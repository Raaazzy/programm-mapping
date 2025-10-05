<template>
  <div class="tabs-wrap">
    <q-tabs
      v-model="activeId"
      inline-label
      shrink
      @update:model-value="goTo"
    >
      <q-tab
        v-for="t in tabs"
        :key="t.id"
        :name="t.id"
        no-caps
        class="saved-tab"
      >
        <div class="row items-center no-wrap">
          <span class="ellipsis">{{ t.title }}</span>

          <q-icon
            name="edit"
            size="16px"
            class="cursor-pointer q-ml-xs text-grey-5 hover-primary"
            @click.stop="openRename(t)"
            :aria-label="`Переименовать ${t.title}`"
          />

          <q-icon
            name="close"
            size="16px"
            class="cursor-pointer q-ml-xs text-grey-5 hover-negative"
            @click.stop="onRemove(t.id)"
            :aria-label="`Закрыть ${t.title}`"
          />
        </div>
      </q-tab>
    </q-tabs>

    <q-dialog v-model="edit.open">
      <q-card style="min-width: 360px; max-width: 90vw">
        <q-card-section class="text-subtitle1">
          Переименовать вкладку
        </q-card-section>
        <q-card-section>
          <q-input
            v-model="edit.title"
            dense
            autofocus
            label="Название"
            @keyup.enter="applyRename"
          />
        </q-card-section>
        <q-card-actions align="right">
          <q-btn flat label="Отмена" v-close-popup />
          <q-btn color="primary" label="Сохранить" @click="applyRename" />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSavedTabs } from '../stores/savedTabs'

const store = useSavedTabs()
const router = useRouter()
const route = useRoute()

onMounted(() => {
  if (typeof store.load === 'function') store.load()
})

const tabs = computed(() => store.tabs)

const activeId = ref(null)
watch(
  [() => route.name, () => route.params.id],
  ([name, id]) => {
    activeId.value = name === 'saved' ? String(id || '') : null
  },
  { immediate: true }
)

function goTo (id) {
  if (!id) return
  router.push({ name: 'saved', params: { id } })
}

function onRemove (id) {
  const isActive = route.name === 'saved' && String(route.params.id) === String(id)
  if (typeof store.remove === 'function') store.remove(id)
  if (isActive) {
    router.push({ name: 'home' })
    activeId.value = null
  }
}

const edit = ref({ open: false, id: null, title: '' })
function openRename (tab) {
  edit.value = { open: true, id: tab.id, title: tab.title || '' }
}
function applyRename () {
  const { id, title } = edit.value
  if (id && typeof store.rename === 'function') {
    store.rename(id, (title || '').trim())
  }
  edit.value.open = false
}
</script>

<style scoped>
.tabs-wrap {
  max-width: 50vw; 
}
.saved-tab :deep(.q-tab__content) {
  gap: 6px; 
}
.ellipsis {
  max-width: 18ch;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.hover-primary:hover {
  color: var(--q-primary) !important;
}
.hover-negative:hover {
  color: var(--q-negative) !important;
}
</style>
