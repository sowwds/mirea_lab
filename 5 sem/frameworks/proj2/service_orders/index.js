const express = require('express');
const cors = require('cors');
const Joi = require('joi');
const app = express();
const PORT = process.env.PORT || 8002;
app.use(cors());
app.use(express.json());
let orders = [];
let currentId = 1;
const orderSchema = Joi.object({
    items: Joi.array().items(Joi.object({
        productId: Joi.number().required(),
        quantity: Joi.number().min(1).required(),
    })).min(1).required(),
});
const checkUserId = (req, res, next) => {
    const userId = req.header('X-User-Id');
    if (!userId) {
        return res.status(401).json({ error: 'Unauthorized: User ID missing.' });
    }
    req.userId = parseInt(userId, 10);
    next();
};
const ordersRouter = express.Router();
app.get('/health', (req, res) => {
    res.json({
        status: 'OK',
        service: 'Orders Service',
        timestamp: new Date().toISOString()
    });
});
ordersRouter.use(checkUserId);
ordersRouter.post('/', (req, res) => {
    const { error } = orderSchema.validate(req.body);
    if (error) {
        return res.status(400).json({ error: error.details[0].message });
    }
    const { items } = req.body;
    const newOrder = {
        id: currentId++,
        userId: req.userId,
        items,
        status: 'created',
        createdAt: new Date().toISOString(),
    };
    orders.push(newOrder);
    res.status(201).json(newOrder);
});
ordersRouter.get('/', (req, res) => {
    const userOrders = orders.filter(o => o.userId === req.userId);
    res.json(userOrders);
});
ordersRouter.get('/:orderId', (req, res) => {
    const orderId = parseInt(req.params.orderId, 10);
    const order = orders.find(o => o.id === orderId);
    if (!order || order.userId !== req.userId) {
        return res.status(404).json({ error: 'Order not found or you do not have permission to view it.' });
    }
    res.json(order);
});
ordersRouter.put('/:orderId', (req, res) => {
    const orderId = parseInt(req.params.orderId, 10);
    const orderIndex = orders.findIndex(o => o.id === orderId && o.userId === req.userId);
    if (orderIndex === -1) {
        return res.status(404).json({ error: 'Order not found or you do not have permission to modify it.' });
    }
    const { status } = req.body;
    if (!status) {
        return res.status(400).json({ error: 'Only status can be updated and is required.' });
    }
    const updatedOrder = { ...orders[orderIndex], status };
    orders[orderIndex] = updatedOrder;
    res.json(updatedOrder);
});
app.use('/orders', ordersRouter);
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Orders service running on port ${PORT}`);
});