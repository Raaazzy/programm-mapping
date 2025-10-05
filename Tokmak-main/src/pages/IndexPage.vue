<template>
  <q-page class="q-pa-md">
    <div class="page-wrap">
      <div class="row q-col-gutter-md">

      <div class="col-12 col-md-3">
        <div class="sticky-col">
          <FiltersPanel v-model="filters" @apply="onApply" />
        </div>
      </div>

      <div class="col-12 col-md-6">
        <div class="center-col">
          <section class="q-pa-md q-gutter-md center-header">
            <SearchWithTags
              v-model="searchQuery"
              v-model:tags="activeTags"
              @submit="onApply"
            />
          </section>

          <div class="center-body">
            <CoursesList
              v-model="selectedCourseIds"
              :filters="filters"
              :search="searchQuery"
              :activeTags="activeTags"
              height="100%"
              @reset="onReset"
            />
          </div>
        </div>
      </div>

      <div class="col-12 col-md-3">
        <div class="sticky-col">
          <RightSelectedCourses
            :selected-ids="selectedCourseIds"
            @remove="onRemove"
          />
        </div>
      </div>

          </div>
    </div>
  </q-page>
</template>

<script setup>
import { ref } from 'vue'
import FiltersPanel from '../components/FiltersPanel.vue'
import CoursesList from '../components/CoursesList.vue'
import RightSelectedCourses from '../components/RightSelectedCourses.vue'
import SearchWithTags from '../components/SearchSection.vue'

const filters = ref({
  faculty: null,
  program: null,
  course: null,
  module: null
})

const selectedCourseIds = ref([])
const searchQuery = ref('')
const activeTags = ref([])

function onApply (payload) {
  
  console.log('apply', payload)
}

function onReset () {
  selectedCourseIds.value = []
}

function onRemove (id) {
  selectedCourseIds.value = selectedCourseIds.value.filter(x => x !== id)
}
</script>

<style scoped>
.sticky-col {
  position: sticky;
  top: 16px;
  max-height: calc(100vh - 32px);
  overflow: auto;
}

.center-col {
  position: sticky;
  top: 16px;
  height: min(85vh, calc(100vh - 32px));
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.center-header {
  flex: 0 0 auto;
  background: var(--q-background, white);
}

.center-body {
  flex: 1 1 auto;
  min-height: 0;
}
.page-wrap {
  margin: 0 auto;
  max-width: 1440px;
  width: 100%;
}
@media (min-width: 1920px) {
  .page-wrap { max-width: 1600px; }
}

</style>
