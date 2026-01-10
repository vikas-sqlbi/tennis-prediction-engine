# Proper Git Merge Procedure

## Standard Workflow for Merging Feature Branches

### Correct Approach (What to do):

1. **Identify the merge conflict location**
   - Feature branch has changes not in dev
   - Dev has changes not in feature branch
   - These changes conflict with each other

2. **Fix conflicts in the feature branch FIRST**
   ```bash
   git checkout feature/branch-name
   git merge dev  # or rebase dev
   ```

3. **Resolve conflicts in feature branch**
   - Edit the files with merge markers
   - Keep both sets of changes if needed
   - Test locally that it works

4. **Commit and push the feature branch**
   ```bash
   git add .
   git commit -m "Resolve merge conflicts: keep both X and Y features"
   git push origin feature/branch-name
   ```

5. **Merge into dev**
   ```bash
   git checkout dev
   git merge feature/branch-name --no-edit
   git push origin dev
   ```

### Why This Matters:

- **Feature branch is the "working copy"**: Conflicts should be resolved where the feature was developed
- **Dev stays clean**: Only accepts clean merges from feature branches
- **Better history**: Commit history shows conflicts were resolved properly in the feature branch
- **Team clarity**: Other developers can see where and how conflicts were fixed

### What NOT to Do:

❌ **DON'T merge first, then fix conflicts in dev**
- This puts temporary conflict markers directly in dev
- Makes dev unstable until fixes are pushed
- Creates confusing commit history

### Example (Feature/Historical-Accuracy-Simulator):

**Wrong way (what I did):**
```
feature/historical-accuracy-simulator
         ↓
    (merge to dev without resolving)
         ↓
dev (BROKEN with conflict markers)
         ↓
    (fix in dev)
         ↓
dev (FIXED, but history is messy)
```

**Right way (what should have been done):**
```
feature/historical-accuracy-simulator
         ↓
    (resolve conflicts here first)
         ↓
feature/historical-accuracy-simulator (CLEAN)
         ↓
    (merge to dev)
         ↓
dev (CLEAN, no conflicts)
```

## Reference for Future Merges

When you see a merge conflict:
1. STOP - don't push yet
2. Go to feature branch
3. Fix it there
4. Push feature branch
5. Merge to dev
