<template>
  <router-view />
</template>

<script setup>
import { useSavedTabs } from 'stores/savedTabs.js'
const store = useSavedTabs()
store.load()
</script>
