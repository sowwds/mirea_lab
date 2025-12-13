import type { Request, Response } from 'express';
const { Router } = require('express');
const db = require('../db');
const jwt = require('jsonwebtoken');
const { authenticateToken } = require('../middleware/auth');

// Define a custom interface for requests that have the user property
interface RequestWithUser extends Request {
  user?: {
    userId: string;
    role: string;
  };
}

const router = Router();
const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-key';

// GET all users (for assignee dropdown)
router.get('/', authenticateToken, async (req: Request, res: Response) => {
    try {
      // Only fetch ID and email, not password or other sensitive info
      const { rows } = await db.query('SELECT id, email FROM "User"'); 
      res.json(rows);
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Failed to fetch users' });
    }
  });

// This route allows a logged-in user to change their own role for development purposes
router.put('/role', authenticateToken, async (req: RequestWithUser, res: Response) => {
  const { role } = req.body;
  const userId = req.user!.userId;

  // Validate the new role
  if (!['ENGINEER', 'MANAGER', 'OBSERVER'].includes(role)) {
    return res.status(400).json({ error: 'Invalid role specified.' });
  }

  try {
    // Update the user's role in the database
    const updateQuery = 'UPDATE "User" SET "role" = $1 WHERE id = $2 RETURNING *';
    const { rows } = await db.query(updateQuery, [role, userId]);
    const updatedUser = rows[0];

    if (!updatedUser) {
      return res.status(404).json({ error: 'User not found.' });
    }

    // Generate a new token with the updated role
    const newToken = jwt.sign({ userId: updatedUser.id, role: updatedUser.role }, JWT_SECRET, {
      expiresIn: '1h',
    });

    res.json({
      message: `Role updated to ${updatedUser.role}`,
      token: newToken, // Send the new token back to the client
    });

  } catch (error) {
    console.error('Failed to update role:', error);
    res.status(500).json({ error: 'An error occurred while updating the role.' });
  }
});

module.exports = router;
