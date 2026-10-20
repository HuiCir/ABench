import unittest

from tools.wrapper_policy import inspect_wrapper


class WrapperPolicyTest(unittest.TestCase):
    digest = "a" * 64

    def wrapper(self, extra: str = "") -> str:
        return (
            "docker run --rm --network none "
            "-v $ABENCH_VIEW_DIR:$ABENCH_VIEW_DIR:ro "
            "-v $ABENCH_WORKSPACE:$ABENCH_WORKSPACE:rw "
            f"{extra} image@sha256:{self.digest}"
        )

    def test_digest_pinned_container_is_accepted(self):
        self.assertTrue(inspect_wrapper(self.wrapper())["scoreable"])

    def test_equivalent_unsafe_options_are_rejected(self):
        for extra in (
            "--env=SECRET=value",
            "--privileged=true",
            "--network host",
            "--volume=/etc:/host:ro",
            "--mount type=bind,src=$ABENCH_VIEW_DIR,dst=/view,ro",
            "--device=/dev/null",
            "--pid host",
            "--cap-add=SYS_ADMIN",
            "--security-opt=seccomp=unconfined",
        ):
            with self.subTest(extra=extra):
                self.assertFalse(inspect_wrapper(self.wrapper(extra))["accepted"])

    def test_workspace_must_be_writable(self):
        raw = self.wrapper().replace(
            "$ABENCH_WORKSPACE:$ABENCH_WORKSPACE:rw",
            "$ABENCH_WORKSPACE:$ABENCH_WORKSPACE:ro",
        )
        self.assertFalse(inspect_wrapper(raw)["accepted"])

    def test_sudo_wrapper_cannot_change_directory_or_preserve_options(self):
        self.assertTrue(inspect_wrapper("sudo -u abench-agent --")["scoreable"])
        self.assertFalse(inspect_wrapper("sudo -u abench-agent -D / --")["accepted"])


if __name__ == "__main__":
    unittest.main()
