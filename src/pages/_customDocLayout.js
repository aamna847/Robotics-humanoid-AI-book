import Layout from '@theme/Layout';
import DocItem from '@theme/DocItem';

export default function CustomDocPage(props) {
  return (
    <Layout>
      <div className="doc-wrapper">
        <DocItem {...props} />
      </div>
    </Layout>
  );
}
