const routes = [
  {
    path: '/',
    component: () => import('../layouts/MainLayout.vue'),
    children: [
      {
        path: '',
        name: 'home',
        component: () => import('../pages/IndexPage.vue'),
      },
      {
        path: 'dashboard',
        name: 'dashboard',
        component: () => import('../pages/DashboardPage.vue'),
      },
      {
        path: 'graph',
        name: 'graph',
        component: () => import('../pages/GraphPage.vue'),
      },
      {
        path: 'cloud',
        name: 'cloud',
        component: () => import('../pages/CloudPage.vue'),
      },
      {
        path: '/saved/:id',
        name: 'saved',
        component: () => import('../pages/SavedViewerPage.vue'),
      },
    ],
  },
]

export default routes
