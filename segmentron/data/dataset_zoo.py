from segmentron.data.transform import Compose, PIPELINES_REGISTRY

def get_dataset_pipelines(pipelines_names):

    pipelines = []
    for pipeline in pipelines_names:
        obj_type = pipeline.pop('type')
        pipelines.append(PIPELINES_REGISTRY.get(obj_type)(**pipeline))
    transform = Compose(pipelines)
    return transform
