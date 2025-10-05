import { createPinia } from 'pinia'

export const pinia = createPinia() // экспортнём инстанс — пригодится вне компонентов

export default ({ app }) => {
  app.use(pinia)
}