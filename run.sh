rmx run birch -f --contain -- 'pip uninstall d4rl -y && python -m scripts.generation.generate_maze2d_datasets && python -m scripts.render_reference'
#rmx run birch -f --contain -- 'pip uninstall d4rl -y  && python -m scripts.render_reference'
#rmx run birch -f --contain -- bash