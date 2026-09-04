# Security

This is a personal project. It runs entirely on your machine, makes no network
calls, and stores no data.

Model files written by `save()` contain only tensors and plain values and are
read back with PyTorch's safe `weights_only` mode. Still, only load model files
you created yourself.

If you find a security problem, please report it privately through GitHub's
"Report a vulnerability" link on the Security tab rather than opening a public
issue.
