const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const Joi = require('joi');
const app = express();
const PORT = process.env.PORT || 8001;
const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-key';
app.use(cors());
app.use(express.json());
let users = [];
let currentId = 1;
const registerSchema = Joi.object({
    name: Joi.string().min(3).required(),
    email: Joi.string().email().required(),
    password: Joi.string().min(6).required(),
});
const loginSchema = Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().required(),
});
app.get('/health', (req, res) => {
    res.json({
        status: 'OK',
        service: 'Users Service',
        timestamp: new Date().toISOString()
    });
});
app.post('/users/register', async (req, res) => {
    const { error } = registerSchema.validate(req.body);
    if (error) {
        return res.status(400).json({ error: error.details[0].message });
    }
    const { name, email, password } = req.body;
    if (users.find(u => u.email === email)) {
        return res.status(409).json({ error: 'User with this email already exists.' });
    }
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    const newUser = {
        id: currentId++,
        name,
        email,
        password: hashedPassword,
        role: 'user', 
    };
    users.push(newUser);
    res.status(201).json({
        id: newUser.id,
        name: newUser.name,
        email: newUser.email,
        role: newUser.role,
    });
});
app.post('/users/login', async (req, res) => {
    const { error } = loginSchema.validate(req.body);
    if (error) {
        return res.status(400).json({ error: error.details[0].message });
    }
    const { email, password } = req.body;
    const user = users.find(u => u.email === email);
    if (!user) {
        return res.status(401).json({ error: 'Invalid credentials.' });
    }
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
        return res.status(401).json({ error: 'Invalid credentials.' });
    }
    const payload = {
        id: user.id,
        role: user.role,
    };
    const token = jwt.sign(payload, JWT_SECRET, { expiresIn: '1h' });
    res.json({ token });
});
app.get('/users/profile', (req, res) => {
    const userId = req.header('X-User-Id');
    if (!userId) {
        return res.status(401).json({ error: 'Not Authorized' });
    }
    const user = users.find(u => u.id === parseInt(userId, 10));
    if (!user) {
        return res.status(404).json({ error: 'User not found' });
    }
    res.json({
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role,
    });
});
app.get('/users/:id', (req, res) => {
    const user = users.find(u => u.id === parseInt(req.params.id, 10));
    if (!user) {
        return res.status(404).json({ error: 'User not found' });
    }
    res.json({
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role,
    });
});
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Users service running on port ${PORT}`);
});