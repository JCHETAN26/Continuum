/**
 * Conventional commits, enforced by the `commit-msg` husky hook.
 *
 * The `type-enum` list is deliberately narrow: the engineering standard names
 * feat/fix/test/docs/refactor, and the remainder cover tooling changes that
 * would otherwise be mislabelled.
 */
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'test', 'docs', 'refactor', 'perf', 'build', 'ci', 'chore', 'revert'],
    ],
    'body-max-line-length': [1, 'always', 100],
    'subject-case': [2, 'never', ['pascal-case', 'upper-case', 'start-case']],
  },
};
