# NAT plugin entry point.
#
# The NeMo Agent Toolkit discovers packages that export the ``nat.components``
# entry-point group (declared in pyproject.toml).  On startup it imports the
# referenced module, which triggers all @register_function decorators defined
# in the guarded ``try / except ImportError`` blocks inside each module.
#
# If ``nvidia-nat`` is NOT installed those blocks are silently skipped and the
# primary ``code-agent run`` CLI path is completely unaffected.
#
# If ``nvidia-nat`` IS installed (pip install "nemo-code-agent[nat]") then:
#   - coder_tool        → registered as _type: coder_tool
#   - read_file_tool    → registered as _type: read_file_tool
#   - execute_bash_tool → registered as _type: execute_bash_tool
#   - workflow          → registered as _type: code_agent_workflow
# and the agent becomes runnable via:  nat run --config configs/config.yml

from nemo_code_agent.tools import coder_tool   # noqa: F401
from nemo_code_agent.tools import filesystem   # noqa: F401
from nemo_code_agent import workflow            # noqa: F401
