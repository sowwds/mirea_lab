import type { Request, Response } from 'express';
const { Router } = require('express');
const db = require('../db');
const { authenticateToken } = require('../middleware/auth');

// Define a custom interface for requests that have the user property
interface RequestWithUser extends Request {
  user?: {
    userId: string;
    role: string;
  };
}

const router = Router();

// Apply authentication middleware to all defect routes
router.use(authenticateToken);

// GET all defects
router.get('/', async (req: Request, res: Response) => {
  try {
    // Note: Double quotes are used for case-sensitive table/column names created by Prisma
    const queryText = `
      SELECT d.*, u.email as "assigneeEmail"
      FROM "Defect" d
      LEFT JOIN "User" u ON d."assigneeId" = u.id
    `;
    const { rows } = await db.query(queryText);
    res.json(rows);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch defects' });
  }
});

// GET a single defect by ID
router.get('/:id', async (req: Request, res: Response) => {
  const { id } = req.params;
  try {
    const queryText = `
      SELECT d.*, u.email as "assigneeEmail"
      FROM "Defect" d
      LEFT JOIN "User" u ON d."assigneeId" = u.id
      WHERE d.id = $1
    `;
    const { rows } = await db.query(queryText, [id]);
    const defect = rows[0];
    if (!defect) {
      return res.status(404).json({ error: 'Defect not found' });
    }
    res.json(defect);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch defect' });
  }
});

// POST (create) a new defect
router.post('/', async (req: RequestWithUser, res: Response) => {
  if (req.user!.role === 'OBSERVER') {
    return res.status(403).json({ error: 'Permission denied. Observers cannot create defects.' });
  }

  const { title, description, priority, status, assigneeId } = req.body;

  if (!title || !assigneeId) {
    return res.status(400).json({ error: 'Title and assigneeId are required' });
  }

  try {
    const queryText = `
      INSERT INTO "Defect" (title, description, priority, status, "assigneeId")
      VALUES ($1, $2, $3, $4, $5)
      RETURNING *
    `;
    const { rows } = await db.query(queryText, [title, description, priority || 'LOW', status || 'NEW', assigneeId]);
    res.status(201).json(rows[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create defect' });
  }
});

// PUT (update) a defect
router.put('/:id', async (req: RequestWithUser, res: Response) => { // Changed req: Request to req: RequestWithUser
  if (req.user!.role === 'OBSERVER') {
    return res.status(403).json({ error: 'Permission denied. Observers cannot update defects.' });
  }

  const { id } = req.params;
  const { title, description, priority, status, assigneeId } = req.body;

  try {
    const queryText = `
      UPDATE "Defect"
      SET title = $1, description = $2, priority = $3, status = $4, "assigneeId" = $5, "updatedAt" = NOW()
      WHERE id = $6
      RETURNING *
    `;
    const { rows } = await db.query(queryText, [title, description, priority, status, assigneeId, id]);
    if (rows.length === 0) {
        return res.status(404).json({ error: 'Defect not found' });
    }
    res.json(rows[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to update defect' });
  }
});

// DELETE a defect
router.delete('/:id', async (req: RequestWithUser, res: Response) => {
  const { id } = req.params; // This line was missing

  if (req.user!.role !== 'MANAGER') {
    return res.status(403).json({ error: 'Permission denied. Only managers can delete defects.' });
  }

  try {
    const queryText = 'DELETE FROM "Defect" WHERE id = $1';
    const result = await db.query(queryText, [id]);
    if (result.rowCount === 0) {
        return res.status(404).json({ error: 'Defect not found' });
    }
    res.status(204).send(); // No Content
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to delete defect' });
  }
});

module.exports = router;
