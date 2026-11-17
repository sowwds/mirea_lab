"""Сервис оформления заказов для практической работы № 5."""


class OrderService:
    """Сохранять и получать заказы через переданные зависимости."""

    def __init__(self, storage, notifier):
        """Принять хранилище заказов и средство отправки уведомлений."""
        self.storage = storage
        self.notifier = notifier

    def place_order(self, order_id, data):
        """Сохранить заказ и уведомить о нём только при успешном сохранении."""
        saved = self.storage.save_order(order_id, data)
        if saved:
            self.notifier.send(order_id, data)
            return True
        return False

    def get_order(self, order_id):
        """Вернуть данные заказа из хранилища или None при его отсутствии."""
        return self.storage.get_order(order_id)
