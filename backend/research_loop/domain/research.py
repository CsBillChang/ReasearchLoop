"""可选研究矩阵参数：表达与证据分离，结构检查不充当科学事实核验。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

DIMENSIONS = ('question', 'method', 'dataset', 'metric', 'limitation')


# 引句只能摘自已返回原文；位置与版本由服务端计算。
class MatrixQuote(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    evidence_id: str = Field(min_length=1, max_length=255)
    quote: str = Field(min_length=8, max_length=1200)


# 不使用approved命名：引句存在不证明其支持整项解读。
class MatrixCell(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    dimension: Literal['question', 'method', 'dataset', 'metric', 'limitation']
    status: Literal['evidenced', 'uncertain', 'conflict', 'missing']
    value: str = Field(default='', max_length=1500)
    conditions: list[str] = Field(default_factory=list, max_length=8)
    reason: str = Field(default='', max_length=1000, description='Required for uncertain, conflict, or missing. Explain the specific uncertainty or checked-scope gap; never infer absence from the whole unread paper.')
    references: list[MatrixQuote] = Field(default_factory=list, max_length=4)

    @model_validator(mode='after')
    def valid_cell(self):
        """缺证留空，其他结论留依据；实验数值显式携带条件。"""
        if self.status == 'missing':
            if self.value or self.references or not self.reason:
                raise ValueError('missing: value/references must be empty; explain the checked-scope gap in reason')
        elif not self.value or not self.references:
            raise ValueError('non-missing cells require value and original references')
        if self.status in ('uncertain', 'conflict') and not self.reason:
            raise ValueError('uncertainty/conflict requires reason')
        if self.status == 'conflict' and len({(q.evidence_id, q.quote) for q in self.references}) < 2:
            raise ValueError('conflict requires two distinct source excerpts')
        if any(not c.strip() or len(c) > 500 for c in self.conditions):
            raise ValueError('conditions must be nonempty and at most 500 characters each')
        if self.dimension == 'metric' and self.status != 'missing' and not self.conditions:
            raise ValueError('metric requires experimental conditions; if unknown use uncertain and explicitly say what is unknown')
        return self


# 一行只属于同一文档版本；来源及paper_id不允许由模型另填。
class MatrixRow(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_id: str = Field(min_length=1, max_length=255)
    inspected_evidence_ids: list[str] = Field(min_length=1, max_length=200)
    cells: list[MatrixCell] = Field(min_length=5, max_length=5)

    @model_validator(mode='after')
    def complete_dimensions(self):
        """五个维度各一项；未取证的维度应明确填missing。"""
        if {c.dimension for c in self.cells} != set(DIMENSIONS):
            raise ValueError('each row requires exactly question/method/dataset/metric/limitation')
        if len(set(self.inspected_evidence_ids)) != len(self.inspected_evidence_ids):
            raise ValueError('duplicate inspected evidence')
        return self


# 乐观版本用于内容变更；同内容的重放由服务端复用。
class EvidenceMatrixInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)
    base_matrix_hash: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    rows: list[MatrixRow] = Field(min_length=1, max_length=8)

    @classmethod
    def describe_validation_error(cls, error, arguments: dict) -> str:
        """为已有结构校验错误附上文档、维度和原参数位置，不改写参数。

        Args:
            error: Pydantic产生的ValidationError。
            arguments: 未通过校验的原始工具参数，只读取定位字段。

        Returns:
            全部错误的可操作说明，序号仍是原始参数的零基序号。
        """
        messages = []
        for item in error.errors(include_url=False):
            loc = item['loc']
            labels = ['.'.join(str(part) for part in loc)]
            try:
                row = arguments['rows'][loc[1]]
                labels.append('document_id=' + str(row.get('document_id', '?')))
                if len(loc) >= 4 and loc[2] == 'cells':
                    cell = row['cells'][loc[3]]
                    labels.append('dimension=' + str(cell.get('dimension', '?')))
            except (KeyError, IndexError, TypeError, AttributeError):
                pass  # 顶层或畸形参数没有可靠定位时，保留Pydantic路径。
            messages.append(' '.join(labels) + ': ' + item['msg'])
        return 'matrix parameter errors:\n' + '\n'.join(messages)

    @model_validator(mode='after')
    def unique_documents(self):
        """同版本只出现一次，不同论文版本可以分别列行。"""
        if len({r.document_id for r in self.rows}) != len(self.rows):
            raise ValueError('duplicate document row')
        return self
