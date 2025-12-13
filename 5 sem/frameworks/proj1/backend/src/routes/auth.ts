import type { Request, Response } from 'express';
const { Router } = require('express');
const db = require('../db');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const router = Router();
const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-key';

// User Registration
router.post('/register', async (req: Request, res: Response) => {
  const { email, password, role } = req.body;

  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required' });
  }

  try {
    const hashedPassword = await bcrypt.hash(password, 10);
    const userRole = role || 'ENGINEER';
    // Note: Table and column names are double-quoted because Prisma creates them with case-sensitivity in PostgreSQL.
    const queryText = 'INSERT INTO "User" (email, password, role) VALUES ($1, $2, $3) RETURNING id';
    const result = await db.query(queryText, [email, hashedPassword, userRole]);

    res.status(201).json({ message: 'User created successfully', userId: result.rows[0].id });
  } catch (error: any) {
    // Check for unique constraint violation (code 23505 for PostgreSQL)
    if (error.code === '23505') {
      return res.status(409).json({ error: 'User with this email already exists' });
    }
    console.error(error);
    res.status(500).json({ error: 'Failed to create user' });
  }
});

// User Login
router.post('/login', async (req: Request, res: Response) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required' });
  }

  try {
    const queryText = 'SELECT * FROM "User" WHERE email = $1';
    const result = await db.query(queryText, [email]);
    const user = result.rows[0];

    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ userId: user.id, role: user.role }, JWT_SECRET, {
      expiresIn: '1h', // Token expires in 1 hour
    });

    res.json({ token });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Login failed' });
  }
});

module.exports = router;
