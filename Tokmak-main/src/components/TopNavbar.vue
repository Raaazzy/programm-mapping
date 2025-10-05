<template>
  <q-header reveal class="no-pointer custom-navbar">
    <q-toolbar class="q-py-sm q-pl-none-md">
      <div class="row items-center home-offset">
        <div class="row items-center">
          <TopNavBarButton
            label="Главная"
            :to="{ name: 'home' }"
            style="font-size: 18px; font-weight: bold; color: white; text-transform: none"
          />
        </div>
      </div>

      <q-space />

      <NavbarSavedTabs />
    </q-toolbar>
  </q-header>
</template>

<script setup>
import TopNavBarButton from '../components/TopNavbarButton.vue'
import NavbarSavedTabs from '../components/SavedNavbarTabs.vue'
</script>


<style lang="scss">
.home-offset {
  margin-left: 18vw;
}

@media (max-width: 2000px) {
  .home-offset {
    margin-left: 0;
  }
}

.no-pointer {
  cursor: default;
}

.custom-navbar {
  color: $secondary;
}
</style>
