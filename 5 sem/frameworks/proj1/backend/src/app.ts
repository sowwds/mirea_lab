import type { Request, Response } from 'express';
const express = require('express');
const cors = require('cors');
require('dotenv').config();

const authRouter = require('./routes/auth');
const defectsRouter = require('./routes/defects');
const usersRouter = require('./routes/users');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.get('/', (req: Request, res: Response) => {
  res.send('Backend is running!');
});
app.use('/api/auth', authRouter);
app.use('/api/defects', defectsRouter);
app.use('/api/users', usersRouter);

module.exports = app;
